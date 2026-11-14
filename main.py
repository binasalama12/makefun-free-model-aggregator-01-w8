#!/usr/bin/env python3
'''One gateway in front of ten free-tier model backends, speaking two client wires.

Clients talk either the OpenAI chat-completions shape or the Anthropic messages
shape, in plain JSON or as server-sent events. Underneath, every call is sent
to whichever free model the brief rates highest among the ones that can meet
the call's needs and are still responding, moving on when a backend turns a
request away with a quota or payment refusal. The gateway also answers the
environment's catalog probe, hands its model list to the environment, works
through the queued workload, and leaves with status zero once told to stop.

Only the standard library is used. Nothing here reads the wall clock to make
a choice: each decision follows from the brief, the incoming call, and what
backends have already answered.
'''

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Per-wallet tuning: this wallet's build pins these values.
# ---------------------------------------------------------------------------
VARIANT_NAME = 'steady-spread'
PREFER_SAME_WIRE_BONUS = 6      # extra milli-rating when a backend speaks the caller's wire
REFUSAL_LIMIT = 4               # quota/payment refusals before a backend is set aside
LONG_CONTEXT_CHARS = 3500       # user text at least this long counts as a long-context call
LONG_CONTEXT_MIN = 200000       # context window a long-context call demands
ANTHROPIC_JSON_SAFE = True      # Anthropic-wire calls are treated as needing JSON support
STREAM_PIECE_CHARS = 8          # re-slice streamed text into pieces this wide; 0 keeps backend pieces
TIE_BREAK = 'brief'             # equal scores: 'brief' keeps directory order, 'alpha' sorts by id

FIXED_CREATED = 1767225600      # fixed creation stamp the simulated backends use
ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
PORT = 8791

# ---------------------------------------------------------------------------
# Tiny HTTP helper built on urllib, with proxies switched off: trials hand
# over plain base URLs, so no proxy should ever get in the way.
# ---------------------------------------------------------------------------
_PLAIN_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_url(method, url, payload=None, extra_headers=None, timeout=15):
    '''Run a single HTTP exchange.

    Hands back a triple of status code, plain header mapping, and raw body
    bytes. A failure to connect at all shows up as status zero with empty
    headers and body, so callers never have to catch anything here.
    '''
    encoded = None
    sent_headers = dict(extra_headers or {})
    if payload is not None:
        encoded = payload if isinstance(payload, bytes) else json.dumps(payload).encode('utf-8')
        sent_headers.setdefault('content-type', 'application/json')
    outgoing = urllib.request.Request(url, data=encoded, headers=sent_headers, method=method)
    try:
        with _PLAIN_OPENER.open(outgoing, timeout=timeout) as reply:
            return reply.status, dict(reply.headers.items()), reply.read()
    except urllib.error.HTTPError as failure:
        heads = dict(failure.headers.items()) if failure.headers else {}
        return failure.code, heads, failure.read()
    except Exception:
        return 0, {}, b''


def fetch_json(method, url, payload=None, extra_headers=None, timeout=15):
    '''Same exchange as fetch_url, but with the body decoded as JSON.'''
    status, heads, raw = fetch_url(method, url, payload, extra_headers, timeout)
    try:
        parsed = json.loads(raw.decode('utf-8')) if raw else None
    except Exception:
        parsed = None
    return status, heads, parsed


# ---------------------------------------------------------------------------
# The brief: provider directory, quality table, and workload list are tucked
# into the prompt file as JSON documents sitting inside its prose.
# ---------------------------------------------------------------------------
def _pull_embedded_doc(text, schema_name):
    '''Fish one JSON document out of the prompt prose by its schema tag.'''
    found = None
    for spelling in ('"schema":"' + schema_name + '"', '"schema": "' + schema_name + '"'):
        spot = text.find(spelling)
        if spot >= 0:
            found = spot
            break
    if found is None:
        return None
    opening = text.rfind('{', 0, found)
    if opening < 0:
        return None
    try:
        doc, _rest = json.JSONDecoder().raw_decode(text[opening:])
        return doc
    except Exception:
        return None


class Upstream:
    '''One backend from the directory: where it lives and how it expects auth.'''

    __slots__ = ('id', 'name', 'wire', 'auth', 'key_env', 'key', 'path')

    def __init__(self, row):
        self.id = row['id']
        self.name = row.get('name', row['id'])
        self.wire = row['wire']
        self.auth = row.get('auth', 'none')
        self.key_env = row.get('key_env', '')
        self.key = row.get('key', '')
        self.path = row['path']

    def endpoint(self):
        return ENV_URL + self.path

    def auth_heads(self):
        if self.auth == 'bearer':
            return {'authorization': 'Bearer ' + self.key}
        if self.auth == 'x-api-key':
            return {'x-api-key': self.key}
        if self.auth == 'x-goog-api-key':
            return {'x-goog-api-key': self.key}
        return {}


def read_brief():
    '''Pull (backends, rating rows, workload entries) from the prompt file.'''
    if not PROMPT_FILE:
        return [], [], []
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as handle:
            prose = handle.read()
    except Exception:
        return [], [], []
    directory = _pull_embedded_doc(prose, 'free-model-aggregator-01/providers/v1') or {}
    ratings_doc = _pull_embedded_doc(prose, 'free-model-aggregator-01/ratings/v1') or {}
    workload_doc = _pull_embedded_doc(prose, 'free-model-aggregator-01/workload/v1') or {}
    backends = [Upstream(row) for row in directory.get('providers', [])]
    return backends, ratings_doc.get('ratings', []), workload_doc.get('manifest', [])


# ---------------------------------------------------------------------------
# The catalog: ask each backend for its own model list first, since that is
# the ground truth, and use the brief's quality table only when a probe
# comes back empty.
# ---------------------------------------------------------------------------
def _fetch_provider_models(backend):
    '''Return [(model id, free?, context window)] from the backend, else None.'''
    if backend.wire == 'gemini':
        target = backend.endpoint() + '/v1beta/models'
    else:
        target = backend.endpoint() + '/v1/models'
    # Slow catalog endpoints need a generous window, so probe at 15 seconds.
    status, _heads, doc = fetch_json('GET', target, extra_headers=backend.auth_heads(), timeout=15)
    if status < 200 or status >= 300 or not isinstance(doc, dict):
        return None
    found = []
    if backend.wire == 'gemini':
        for entry in doc.get('models', []):
            mid = str(entry.get('name', '')).removeprefix('models/')
            if mid:
                window = entry.get('context_length') or entry.get('inputTokenLimit')
                found.append((mid, entry.get('free') is True, window))
    else:
        for entry in doc.get('data', []):
            if isinstance(entry.get('id'), str) and entry['id']:
                found.append((entry['id'], entry.get('free') is True, entry.get('context_length')))
    return found or None


def assemble_catalog(backends, rating_rows):
    '''Build the catalog the environment is shown at registration time.'''
    fallback = {}
    for row in rating_rows:
        fallback.setdefault(row['provider'], []).append(
            (row['model'], row.get('free') is True, row.get('context_length')))
    catalog = {'providers': []}
    for backend in backends:
        probed = _fetch_provider_models(backend) if ENV_URL else None
        if probed is None:
            probed = fallback.get(backend.id, [])
        catalog['providers'].append({
            'id': backend.id,
            'base_url': backend.path,
            'key_env': backend.key_env,
            'wire': backend.wire,
            'models': [
                {'id': mid, 'free': is_free, 'context_length': window}
                for mid, is_free, window in probed
            ],
        })
    return catalog


def catalog_json(catalog):
    return json.dumps(catalog)


# ---------------------------------------------------------------------------
# Canonical call: the client's request reshaped into one wire-neutral form,
# keeping every part of it — nothing is dropped on the way in.
# ---------------------------------------------------------------------------
def _text_from_blocks(content):
    '''Collect the plain text out of a string or a list of content blocks.'''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                piece = block.get('text')
                if isinstance(piece, str):
                    chunks.append(piece)
        return '\n'.join(chunks)
    return ''


def _image_part(url):
    '''Turn a data: URL into an inline image part, anything else into a reference.'''
    if isinstance(url, str) and url.startswith('data:') and ';base64,' in url:
        media = url[5:url.index(';base64,')]
        payload = url.split(';base64,', 1)[1]
        return {'type': 'image', 'media_type': media, 'data': payload}
    return {'type': 'image_ref', 'url': url}


class ClientRequest:
    '''Wire-neutral call.

    system holds the system text or None; messages is a list of {role, parts}
    where each part is {type: text, text}, {type: image, media_type, data}, or
    {type: image_ref, url}; tools is a list of {name, description,
    parameters}; raw keeps the client's original body untouched.
    '''

    def __init__(self, api, body):
        self.api = api
        self.raw = body
        self.stream = body.get('stream') is True
        self.max_tokens = None
        self.stops = []
        self.json_mode = False
        self.response_format = None
        self.system = None
        self.messages = []
        self.tools = []
        self.tool_choice = None
        if api == 'openai':
            self._ingest_openai(body)
        else:
            self._ingest_anthropic(body)

    def _ingest_openai(self, body):
        limit = body.get('max_tokens')
        if not isinstance(limit, (int, float)):
            limit = body.get('max_completion_tokens')
        self.max_tokens = limit if isinstance(limit, (int, float)) else None
        stop = body.get('stop')
        if isinstance(stop, str):
            self.stops = [stop]
        elif isinstance(stop, list):
            self.stops = [entry for entry in stop if isinstance(entry, str)]
        fmt = body.get('response_format')
        if isinstance(fmt, dict):
            self.response_format = fmt
            self.json_mode = fmt.get('type') in ('json_object', 'json_schema')
        self.tool_choice = body.get('tool_choice')
        for tool in body.get('tools') or []:
            fn = tool.get('function') if isinstance(tool, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get('name'), str):
                self.tools.append({
                    'name': fn['name'],
                    'description': fn.get('description', ''),
                    'parameters': fn.get('parameters') or {'type': 'object'},
                })
        system_chunks = []
        for msg in body.get('messages') or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                text = _text_from_blocks(content)
                if text:
                    system_chunks.append(text)
                continue
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({'type': 'text', 'text': content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get('type') == 'text' and isinstance(block.get('text'), str):
                        parts.append({'type': 'text', 'text': block['text']})
                    elif block.get('type') == 'image_url' and isinstance(block.get('image_url'), dict):
                        parts.append(_image_part(block['image_url'].get('url', '')))
            self.messages.append({'role': role or 'user', 'parts': parts})
        self.system = '\n'.join(system_chunks) if system_chunks else None

    def _ingest_anthropic(self, body):
        limit = body.get('max_tokens')
        self.max_tokens = limit if isinstance(limit, (int, float)) else None
        stops = body.get('stop_sequences')
        if isinstance(stops, list):
            self.stops = [entry for entry in stops if isinstance(entry, str)]
        self.tool_choice = body.get('tool_choice')
        system = body.get('system')
        if isinstance(system, str):
            self.system = system or None
        elif isinstance(system, list):
            text = _text_from_blocks(system)
            self.system = text or None
        for tool in body.get('tools') or []:
            if isinstance(tool, dict) and isinstance(tool.get('name'), str):
                self.tools.append({
                    'name': tool['name'],
                    'description': tool.get('description', ''),
                    'parameters': tool.get('input_schema') or {'type': 'object'},
                })
        for msg in body.get('messages') or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get('content')
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({'type': 'text', 'text': content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get('type') == 'text' and isinstance(block.get('text'), str):
                        parts.append({'type': 'text', 'text': block['text']})
                    elif block.get('type') == 'image' and isinstance(block.get('source'), dict):
                        src = block['source']
                        if src.get('type') == 'base64' and isinstance(src.get('data'), str):
                            parts.append({'type': 'image', 'media_type': src.get('media_type', 'image/png'), 'data': src['data']})
                        elif src.get('type') == 'url':
                            parts.append({'type': 'image_ref', 'url': src.get('url', '')})
            self.messages.append({'role': msg.get('role') or 'user', 'parts': parts})

    # -- what the call needs, read straight off the call itself -------------
    @property
    def wants_tools(self):
        return len(self.tools) > 0

    @property
    def wants_vision(self):
        for msg in self.messages:
            for part in msg['parts']:
                if part['type'] in ('image', 'image_ref'):
                    return True
        return False

    @property
    def latest_user_text(self):
        for msg in reversed(self.messages):
            if msg['role'] == 'user':
                return '\n'.join(part['text'] for part in msg['parts'] if part['type'] == 'text')
        return ''


# ---------------------------------------------------------------------------
# Turning the canonical call back into a backend-wire request.
# ---------------------------------------------------------------------------
def _as_openai_messages(req):
    shaped = []
    if req.system is not None:
        shaped.append({'role': 'system', 'content': req.system})
    for msg in req.messages:
        parts = msg['parts']
        if all(part['type'] == 'text' for part in parts):
            joined = '\n'.join(part['text'] for part in parts)
            shaped.append({'role': msg['role'], 'content': joined})
            continue
        content = []
        for part in parts:
            if part['type'] == 'text':
                content.append({'type': 'text', 'text': part['text']})
            elif part['type'] == 'image':
                content.append({'type': 'image_url', 'image_url': {
                    'url': 'data:%s;base64,%s' % (part['media_type'], part['data'])}})
            else:
                content.append({'type': 'image_url', 'image_url': {'url': part['url']}})
        shaped.append({'role': msg['role'], 'content': content})
    return shaped


def _as_anthropic_messages(req):
    shaped = []
    for msg in req.messages:
        parts = msg['parts']
        if all(part['type'] == 'text' for part in parts):
            joined = '\n'.join(part['text'] for part in parts)
            shaped.append({'role': msg['role'], 'content': joined})
            continue
        content = []
        for part in parts:
            if part['type'] == 'text':
                content.append({'type': 'text', 'text': part['text']})
            elif part['type'] == 'image':
                content.append({'type': 'image', 'source': {
                    'type': 'base64', 'media_type': part['media_type'], 'data': part['data']}})
            else:
                content.append({'type': 'image', 'source': {'type': 'url', 'url': part['url']}})
        shaped.append({'role': msg['role'], 'content': content})
    return shaped


def _as_gemini_payload(req):
    contents = []
    for msg in req.messages:
        parts = []
        for part in msg['parts']:
            if part['type'] == 'text':
                parts.append({'text': part['text']})
            elif part['type'] == 'image':
                parts.append({'inlineData': {'mimeType': part['media_type'], 'data': part['data']}})
            else:
                parts.append({'fileData': {'fileUri': part['url']}})
        role = 'model' if msg['role'] == 'assistant' else 'user'
        contents.append({'role': role, 'parts': parts})
    payload = {'contents': contents}
    if req.system is not None:
        payload['systemInstruction'] = {'parts': [{'text': req.system}]}
    if req.tools:
        payload['tools'] = [{'functionDeclarations': [
            {'name': tool['name'], 'description': tool['description'], 'parameters': tool['parameters']}
            for tool in req.tools]}]
    config = {}
    if req.max_tokens is not None:
        config['maxOutputTokens'] = req.max_tokens
    if req.stops:
        config['stopSequences'] = list(req.stops)
    if req.json_mode:
        config['responseMimeType'] = 'application/json'
    if config:
        payload['generationConfig'] = config
    return payload


def _convert_tool_choice(wire, req):
    '''Translate the client's tool_choice onto a backend wire.

    Returns (translated choice, whether the backend should receive one).
    '''
    choice = req.tool_choice
    if choice is None:
        return None, False
    if wire == 'openai':
        return choice, True
    if wire == 'anthropic':
        if isinstance(choice, str):
            picked = choice if choice in ('auto', 'any', 'none') else 'auto'
            return {'type': picked}, True
        if isinstance(choice, dict):
            if choice.get('type') == 'function' and isinstance(choice.get('function'), dict):
                return {'type': 'tool', 'name': choice['function'].get('name')}, True
            if choice.get('type') in ('auto', 'any', 'none'):
                return {'type': choice['type']}, True
        return None, False
    return None, False


def build_upstream_call(backend, model_id, req):
    '''Build (url, headers, body) for one inference call on a backend.'''
    headers = backend.auth_heads()
    if backend.wire == 'openai':
        if req.api == 'openai':
            body = json.loads(json.dumps(req.raw))  # client's own body, passed along as-is
            body['model'] = model_id
        else:
            body = {'model': model_id, 'messages': _as_openai_messages(req)}
            if req.max_tokens is not None:
                body['max_tokens'] = req.max_tokens
            if req.stops:
                body['stop'] = list(req.stops)
            if req.tools:
                body['tools'] = [{'type': 'function', 'function': {
                    'name': tool['name'], 'description': tool['description'],
                    'parameters': tool['parameters']}} for tool in req.tools]
            translated, usable = _convert_tool_choice('openai', req)
            if usable:
                body['tool_choice'] = translated
            if req.stream:
                body['stream'] = True
        if req.stream:
            body['stream'] = True
            stream_opts = dict(body.get('stream_options') or {})
            stream_opts['include_usage'] = True
            body['stream_options'] = stream_opts
        return backend.endpoint() + '/v1/chat/completions', headers, body
    if backend.wire == 'anthropic':
        headers['anthropic-version'] = '2023-06-01'
        if req.api == 'anthropic':
            body = json.loads(json.dumps(req.raw))
            body['model'] = model_id
        else:
            body = {'model': model_id, 'messages': _as_anthropic_messages(req)}
            if req.system is not None:
                body['system'] = req.system
            if req.max_tokens is not None:
                body['max_tokens'] = req.max_tokens
            if req.stops:
                body['stop_sequences'] = list(req.stops)
            if req.tools:
                body['tools'] = [{'name': tool['name'], 'description': tool['description'],
                                  'input_schema': tool['parameters']} for tool in req.tools]
            translated, usable = _convert_tool_choice('anthropic', req)
            if usable:
                body['tool_choice'] = translated
            if req.stream:
                body['stream'] = True
        return backend.endpoint() + '/v1/messages', headers, body
    # gemini
    body = _as_gemini_payload(req)
    verb = 'streamGenerateContent' if req.stream else 'generateContent'
    url = backend.endpoint() + '/v1beta/models/%s:%s' % (
        urllib.parse.quote(model_id, safe=''), verb)
    if req.stream:
        url += '?alt=sse'
    return url, headers, body


# ---------------------------------------------------------------------------
# Reading a backend's answer into (text, pieces, stop reason, usage in/out).
# ---------------------------------------------------------------------------
def _split_sse_frames(raw):
    '''Split a server-sent-events body into (event name, data text) pairs.'''
    frames = []
    normalized = raw.decode('utf-8', 'replace').replace('\r\n', '\n')
    for block in normalized.split('\n\n'):
        event = None
        data_lines = []
        for line in block.split('\n'):
            if line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                data_lines.append(line[5:].lstrip(' '))
        if data_lines:
            frames.append((event, '\n'.join(data_lines)))
    return frames


def interpret_answer(wire, raw, stream):
    '''Decode a backend reply into {text, pieces, stop, usage} or None.'''
    try:
        if wire == 'openai':
            if not stream:
                doc = json.loads(raw.decode('utf-8'))
                choice = doc['choices'][0]
                content = choice['message']['content']
                if isinstance(content, list):
                    content = ''.join(block.get('text', '') for block in content if isinstance(block, dict))
                usage = doc.get('usage') or {}
                return {'text': content, 'pieces': [content], 'stop': choice.get('finish_reason'),
                        'usage': (usage.get('prompt_tokens'), usage.get('completion_tokens'))}
            text, pieces, stop, usage = '', [], None, None
            for _event, data in _split_sse_frames(raw):
                if data.strip() == '[DONE]':
                    continue
                chunk = json.loads(data)
                if isinstance(chunk.get('usage'), dict):
                    usage = (chunk['usage'].get('prompt_tokens'), chunk['usage'].get('completion_tokens'))
                if chunk.get('choices'):
                    choice = chunk['choices'][0]
                    delta = choice.get('delta') or {}
                    if isinstance(delta.get('content'), str) and delta['content']:
                        pieces.append(delta['content'])
                        text += delta['content']
                    if isinstance(choice.get('finish_reason'), str):
                        stop = choice['finish_reason']
            return {'text': text, 'pieces': pieces, 'stop': stop, 'usage': usage}
        if wire == 'anthropic':
            if not stream:
                doc = json.loads(raw.decode('utf-8'))
                text = ''.join(block.get('text', '') for block in doc.get('content', [])
                               if isinstance(block, dict) and block.get('type') == 'text')
                usage = doc.get('usage') or {}
                return {'text': text, 'pieces': [text], 'stop': doc.get('stop_reason'),
                        'usage': (usage.get('input_tokens'), usage.get('output_tokens'))}
            text, pieces, stop = '', [], None
            in_tok, out_tok = None, None
            for event, data in _split_sse_frames(raw):
                doc = json.loads(data)
                if event == 'message_start':
                    in_tok = (doc.get('message') or {}).get('usage', {}).get('input_tokens')
                elif event == 'content_block_delta':
                    delta = doc.get('delta') or {}
                    if delta.get('type') == 'text_delta' and isinstance(delta.get('text'), str):
                        pieces.append(delta['text'])
                        text += delta['text']
                elif event == 'message_delta':
                    stop = (doc.get('delta') or {}).get('stop_reason') or stop
                    out_tok = (doc.get('usage') or {}).get('output_tokens', out_tok)
            usage = (in_tok, out_tok) if in_tok is not None or out_tok is not None else None
            return {'text': text, 'pieces': pieces, 'stop': stop, 'usage': usage}
        # gemini
        if not stream:
            doc = json.loads(raw.decode('utf-8'))
            cand = doc['candidates'][0]
            text = ''.join(part.get('text', '') for part in cand['content']['parts'] if isinstance(part, dict))
            meta = doc.get('usageMetadata') or {}
            return {'text': text, 'pieces': [text], 'stop': cand.get('finishReason'),
                    'usage': (meta.get('promptTokenCount'), meta.get('candidatesTokenCount'))}
        text, pieces, stop, usage = '', [], None, None
        for _event, data in _split_sse_frames(raw):
            doc = json.loads(data)
            for cand in doc.get('candidates', []):
                for part in (cand.get('content') or {}).get('parts', []):
                    if isinstance(part, dict) and isinstance(part.get('text'), str) and part['text']:
                        pieces.append(part['text'])
                        text += part['text']
                if isinstance(cand.get('finishReason'), str):
                    stop = cand['finishReason']
            meta = doc.get('usageMetadata')
            if isinstance(meta, dict):
                usage = (meta.get('promptTokenCount'), meta.get('candidatesTokenCount'))
        return {'text': text, 'pieces': pieces, 'stop': stop, 'usage': usage}
    except Exception:
        return None


STOP_MAP_OPENAI = {'stop': 'stop', 'end_turn': 'stop', 'STOP': 'stop',
                   'max_tokens': 'length', 'MAX_TOKENS': 'length',
                   'tool_use': 'tool_calls', 'refusal': 'content_filter', 'SAFETY': 'content_filter'}
STOP_MAP_ANTHROPIC = {'stop': 'end_turn', 'STOP': 'end_turn', 'end_turn': 'end_turn',
                      'length': 'max_tokens', 'MAX_TOKENS': 'max_tokens',
                      'tool_calls': 'tool_use', 'function_call': 'tool_use',
                      'content_filter': 'refusal', 'SAFETY': 'refusal'}


def _stream_pieces(answer):
    '''Pieces to emit while streaming: re-sliced evenly, or the raw pieces.'''
    kept = [piece for piece in answer['pieces'] if piece]
    if STREAM_PIECE_CHARS > 0:
        full = answer['text']
        width = STREAM_PIECE_CHARS
        kept = [full[start:start + width] for start in range(0, len(full), width)] or ['']
    return kept or ['']


# ---------------------------------------------------------------------------
# Answering the client back on its own wire, streamed exactly as asked.
# ---------------------------------------------------------------------------
def render_client_reply(req, model_id, answer):
    '''Build (content type, body bytes) for the client's wire.'''
    usage = answer['usage'] or (None, None)
    if req.api == 'openai':
        stop = STOP_MAP_OPENAI.get(answer['stop'], 'stop')
        if not req.stream:
            doc = {'id': 'chatcmpl-agg', 'object': 'chat.completion', 'created': FIXED_CREATED,
                   'model': model_id,
                   'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer['text']},
                                'finish_reason': stop, 'logprobs': None}],
                   'usage': {'prompt_tokens': usage[0], 'completion_tokens': usage[1],
                             'total_tokens': (usage[0] or 0) + (usage[1] or 0)}}
            return 'application/json', json.dumps(doc).encode('utf-8')
        frames = []

        def make_chunk(delta, finish):
            return {'id': 'chatcmpl-agg', 'object': 'chat.completion.chunk', 'created': FIXED_CREATED,
                    'model': model_id,
                    'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}

        frames.append('data: ' + json.dumps(make_chunk({'role': 'assistant', 'content': ''}, None)) + '\n\n')
        for piece in _stream_pieces(answer):
            frames.append('data: ' + json.dumps(make_chunk({'content': piece}, None)) + '\n\n')
        frames.append('data: ' + json.dumps(make_chunk({}, stop)) + '\n\n')
        if usage[0] is not None:
            frames.append('data: ' + json.dumps({
                'id': 'chatcmpl-agg', 'object': 'chat.completion.chunk', 'created': FIXED_CREATED,
                'model': model_id, 'choices': [],
                'usage': {'prompt_tokens': usage[0], 'completion_tokens': usage[1],
                          'total_tokens': (usage[0] or 0) + (usage[1] or 0)}}) + '\n\n')
        frames.append('data: [DONE]\n\n')
        return 'text/event-stream', ''.join(frames).encode('utf-8')
    # anthropic
    stop = STOP_MAP_ANTHROPIC.get(answer['stop'], 'end_turn')
    if not req.stream:
        doc = {'id': 'msg_agg', 'type': 'message', 'role': 'assistant', 'model': model_id,
               'content': [{'type': 'text', 'text': answer['text']}],
               'stop_reason': stop, 'stop_sequence': None,
               'usage': {'input_tokens': usage[0], 'output_tokens': usage[1]}}
        return 'application/json', json.dumps(doc).encode('utf-8')
    frames = []

    def make_frame(event, doc):
        return 'event: %s\ndata: %s\n\n' % (event, json.dumps(doc))

    frames.append(make_frame('message_start', {
        'type': 'message_start',
        'message': {'id': 'msg_agg', 'type': 'message', 'role': 'assistant', 'model': model_id,
                    'content': [], 'stop_reason': None, 'stop_sequence': None,
                    'usage': {'input_tokens': usage[0], 'output_tokens': 0}}}))
    frames.append(make_frame('content_block_start', {
        'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}))
    for piece in _stream_pieces(answer):
        frames.append(make_frame('content_block_delta', {
            'type': 'content_block_delta', 'index': 0,
            'delta': {'type': 'text_delta', 'text': piece}}))
    frames.append(make_frame('content_block_stop', {'type': 'content_block_stop', 'index': 0}))
    frames.append(make_frame('message_delta', {
        'type': 'message_delta',
        'delta': {'stop_reason': stop, 'stop_sequence': None},
        'usage': {'output_tokens': usage[1]}}))
    frames.append(make_frame('message_stop', {'type': 'message_stop'}))
    return 'text/event-stream', ''.join(frames).encode('utf-8')


# ---------------------------------------------------------------------------
# Picking a backend: rank the free models that can handle the call by the
# brief's rating, try them in order, and keep count of refusals as they come.
# ---------------------------------------------------------------------------
class Picker:
    def __init__(self, backends, rating_rows):
        self.backends = {backend.id: backend for backend in backends}
        self.order = [backend.id for backend in backends]
        self.rows = {}
        for row in rating_rows:
            self.rows[(row['provider'], row['model'])] = row
        self.spent = {pid: 0 for pid in self.order}
        self.refusals = {pid: 0 for pid in self.order}
        self.exhausted = set()

    def ranked(self, req):
        want_json = req.json_mode or (req.api == 'anthropic' and ANTHROPIC_JSON_SAFE)
        long_call = len(req.latest_user_text) >= LONG_CONTEXT_CHARS
        scored = []
        for position, pid in enumerate(self.order):
            backend = self.backends[pid]
            for (prov_id, model_id), row in self.rows.items():
                if prov_id != pid or row.get('free') is not True:
                    continue
                if req.wants_tools and not row.get('tools'):
                    continue
                if req.wants_vision and not row.get('vision'):
                    continue
                if want_json and not row.get('json'):
                    continue
                if long_call and (row.get('context_length') or 0) < LONG_CONTEXT_MIN:
                    continue
                rating = row.get('rating_milli', 0)
                if backend.wire == req.api:
                    rating += PREFER_SAME_WIRE_BONUS
                scored.append((-rating, position, backend, model_id))
        if TIE_BREAK == 'alpha':
            scored.sort(key=lambda entry: (entry[0], entry[2].id, entry[3]))
        else:
            scored.sort(key=lambda entry: (entry[0], entry[1]))
        return [(backend, model_id) for _neg, _pos, backend, model_id in scored]

    def dispatch(self, req):
        for backend, model_id in self.ranked(req):
            if backend.id in self.exhausted:
                continue
            url, headers, body = build_upstream_call(backend, model_id, req)
            status, _heads, raw = fetch_url('POST', url, body, headers, timeout=20)
            self.spent[backend.id] += 1
            if 200 <= status < 300:
                answer = interpret_answer(backend.wire, raw, req.stream)
                if answer is not None:
                    return model_id, answer
                continue
            if status in (402, 429):
                self.refusals[backend.id] += 1
                if self.refusals[backend.id] >= REFUSAL_LIMIT:
                    self.exhausted.add(backend.id)
                continue
            # Anything else (400, 5xx, no answer at all): try the next model.
        return None


# ---------------------------------------------------------------------------
# The server the environment's client talks to.
# ---------------------------------------------------------------------------
RUNTIME = {'picker': None, 'catalog': None}


def list_models(headers):
    catalog = RUNTIME['catalog'] or {'providers': []}
    ids = []
    for provider in catalog['providers']:
        for model in provider['models']:
            if model['free'] and model['id'] not in ids:
                ids.append(model['id'])
    if headers.get('x-api-key') is not None:
        doc = {'data': [{'type': 'model', 'id': mid, 'display_name': mid,
                         'created_at': '2026-01-01T00:00:00Z'} for mid in ids],
               'has_more': False,
               'first_id': ids[0] if ids else None,
               'last_id': ids[-1] if ids else None}
    else:
        doc = {'object': 'list',
               'data': [{'id': mid, 'object': 'model', 'created': FIXED_CREATED,
                         'owned_by': 'aggregator'} for mid in ids]}
    return json.dumps(doc).encode('utf-8')


def failure_payload(api, message):
    if api == 'anthropic':
        return json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': message}}).encode('utf-8')
    return json.dumps({'error': {'message': message, 'type': 'api_error',
                                 'param': None, 'code': 'no_provider'}}).encode('utf-8')


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, body):
        try:
            self.send_response(status)
            self.send_header('content-type', content_type)
            self.send_header('content-length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.split('?')[0] == '/v1/models':
            self._send(200, 'application/json', list_models(self.headers))
        else:
            self._send(404, 'application/json', failure_payload('openai', 'not found'))

    def do_POST(self):
        path = self.path.split('?')[0]
        if path == '/v1/chat/completions':
            api = 'openai'
        elif path == '/v1/messages':
            api = 'anthropic'
        else:
            self._send(404, 'application/json', failure_payload('openai', 'not found'))
            return
        try:
            length = int(self.headers.get('content-length') or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b''
        try:
            body = json.loads(raw.decode('utf-8'))
            if not isinstance(body, dict):
                raise ValueError('body is not an object')
        except Exception:
            self._send(400, 'application/json', failure_payload(api, 'the request body is not one JSON object'))
            return
        req = ClientRequest(api, body)
        served = RUNTIME['picker'].dispatch(req)
        if served is None:
            self._send(502, 'application/json', failure_payload(api, 'no free model could serve this request'))
            return
        model_id, answer = served
        content_type, payload = render_client_reply(req, model_id, answer)
        self._send(200, content_type, payload)


# ---------------------------------------------------------------------------
# Trial lifecycle: probe the catalog, register with the environment, serve
# until the environment says it is done, then exit zero.
# ---------------------------------------------------------------------------
def run(argv):
    backends, ratings, _manifest = read_brief()
    if '--list' in argv:
        if not backends:
            print('{"providers":[]}')
            return 0
        print(catalog_json(assemble_catalog(backends, ratings)))
        return 0
    catalog = assemble_catalog(backends, ratings)
    RUNTIME['catalog'] = catalog
    RUNTIME['picker'] = Picker(backends, ratings)
    server = ThreadingHTTPServer(('0.0.0.0', PORT), GatewayHandler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05})
    worker.daemon = True
    worker.start()
    if ENV_URL:
        # Register until the environment acknowledges, or the tries run out;
        # a slow environment start must not end the run before it begins.
        for _attempt in range(30):
            status, _heads, doc = fetch_json('POST', ENV_URL + '/register',
                                             {'protocol_version': 1, 'port': PORT, 'catalog': catalog},
                                             timeout=5)
            if status == 200 and isinstance(doc, dict) and doc.get('ok') is True:
                break
            time.sleep(0.5)
        misses = 0
        waited = 0.0
        while waited < 170.0:
            status, _heads, doc = fetch_json('GET', ENV_URL + '/work', timeout=5)
            if status == 200 and isinstance(doc, dict):
                misses = 0
                if doc.get('done') is True:
                    break
            else:
                misses += 1
                if misses > 40:
                    break
            time.sleep(0.2)
            waited += 0.2
    else:
        worker.join()
    server.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(run(sys.argv[1:]))
