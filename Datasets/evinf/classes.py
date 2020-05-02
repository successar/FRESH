from collections import defaultdict
from itertools import groupby

import spacy
NLP = spacy.load("en_core_web_sm")

import tools

def string_to_tokens(string):
    nlp = NLP(string)
    return [Span(t.idx, t.idx+len(t.text), t.text) for t in nlp]

class Span:
    def __init__(self, i, f, text, label = None, concepts = None):
        self.i = i
        self.f = f
        self.text = text
        self.label = label
        self.concepts = concepts 

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        label_str = ' ({})'.format(self.label) if self.label != None else ''
        s = '[{}:{} "{}"{}]'.format(self.i, self.f, self.text.replace('\n', '\\n'), label_str)
        return s

class Entity:
    def __init__(self, entity_string):
        self.string = entity_string
        self.type = None
        self.name = None
        self.concepts = None
        self.mentions = None

    def __str__(self):
        return '[ENTITIY: {}]'.format(self.string)

    def set_name(self):
        assert self.mentions != None
        concept_counts = { c: 0 for c in self.concepts }
        for mention in self.mentions:
            for concept in mention.concepts:
                if concept in concept_counts:
                    concept_counts[concept] += 1
        self.name = max(concept_counts, key = concept_counts.get)


class Frame:
    label_encoder = { \
        'No significant difference': 0,
        'Significantly increased': 1,
        'Significantly decreased': 2,
        'no significant difference': 0,
        'significantly increased': 1,
        'significantly decreased': 2
    }

    encoder_label = {
        v:k for k,v  in label_encoder.items()
    }

    def __init__(self, i, c, o, ev, label):
        self.i = i
        self.c = c
        self.o = o
        self.ev = ev

        self.label = None
        if type(label) == str:
            if label in self.label_encoder:
                self.label = self.label_encoder[label]
            elif label.isdigit():
                self.label = int(label)
        elif type(label) == int and label in self.label_encoder.values():
            self.label = label

        if self.label == None:
            raise Exception('Unable to parse label for frame: {}'.format(label))

    def __str__(self):
        return '[I: {}, C: {}, O: {}, label: {}]'.format(self.i, self.c, self.o, self.label)


class Doc:
    def __init__(self, d_id, text):
        self.id = d_id
        # everything that references text offsets need to go here:
        self.char_labels = defaultdict(list)
        self.token_labels = defaultdict(list)
        self.text = text
        self.frames = []
        self.coref_groups = []
        self.parsed = False

    @classmethod
    def init_from_tokens(cls, d_id, tokens):
        text = ''
        spans = []
        i = 0
        for t in tokens:
            if i > 0:
                text += ' '; i += 1
            f = i + len(t)
            spans.append(Span(i, f, t))
            text += t; i = f
        doc = cls(d_id, text)
        doc.source_spans = spans
        return doc

    @classmethod
    def init_from_text(cls, d_id, text):
        doc = cls(d_id, text)
        return doc

    @classmethod
    def load_from_file(cls, doc_id):
        pass

    def write_to_file(self):
        pass

    def parse_text(self):
        nlp = NLP(self.text)
        self.tokens = [Span(t.idx, t.idx+len(t.text), t.text) for t in nlp]
        self.sents = [Span(s.start_char, s.end_char, s.text) for s in nlp.sents]
        self.parsed = True

    def add_coref_groups(self, spans):
        key_fn = lambda e: e.label
        sorted_spans = sorted(spans, key = key_fn)
        for label, label_spans in groupby(sorted_spans, key = key_fn):
            self.coref_groups.append(list(label_spans))
        for s in spans:
            self.char_labels['coref_spans'].append(s)

    def substitute_string(self, start_str, substitutions):
        new_str = ''
        prev_end = 0
        char_offsets = [0]*len(start_str)
        for start, end, text in substitutions:
            new_str += start_str[prev_end:start]
            new_str += text
            prev_end = end
            len_delta = len(text) - (end - start)
            for i in range(end, len(start_str)):
                char_offsets[i] += len_delta
        new_str += start_str[prev_end:]
        return new_str, char_offsets

    def get_sf_token_substitutions(self, string, tokens = None):
        tokens = tokens or string_to_tokens(string)
        text_substitutions = []
        for token in tokens:
            if token.text in self.sf_lf_map:
                text_substitutions.append((token.i, token.f, self.sf_lf_map[token.text]))
        return text_substitutions

    def get_sf_substituted_string(self, string):
        subs = self.get_sf_token_substitutions(string)
        new_str, char_offsets = self.substitute_string(string, subs)
        return new_str

    def update_text(self, substitutions):
        new_text, char_offsets = self.substitute_string(self.text, substitutions)
        self.text = new_text
        for label_class in self.char_labels:
            for span in self.char_labels[label_class]:
                if span.i >= 0:
                    span.i += char_offsets[span.i]
                    span.f += char_offsets[span.f]
                    span.text = self.text[span.i:span.f]
        self.parse_text()

    def replace_acronyms(self):
        self.sf_lf_map = tools.ab3p_text(self.text)
        if not self.parsed:
            self.parse_text()
        text_substitutions = self.get_sf_token_substitutions(self.text, self.tokens)
        self.update_text(text_substitutions)
        self.replace_frame_acronyms()

    def replace_frame_acronyms(self):
        for frame in self.frames:
            for sf, lf in self.sf_lf_map.items():
                frame.i = self.get_sf_substituted_string(frame.i)
                frame.c = self.get_sf_substituted_string(frame.c)
                frame.o = self.get_sf_substituted_string(frame.o)

    def metamap_text(self):
        self.metamap_spans = []
        for phrase in tools.get_mm_phrases(self.text):
            span = Span(phrase.i, phrase.f, phrase.text, concepts = phrase.concepts)
            self.metamap_spans.append(span)

    def metamap_frames(self):
        for frame in self.frames:
            for e in 'ico':
                e_str = getattr(frame, e)
                e_concepts = set()
                for phrase in tools.get_mm_phrases(e_str):
                    e_concepts.update(phrase.concepts)
                setattr(frame, 'metamap_'+e, Span(-1, -1, e_str, concepts = e_concepts)) 
