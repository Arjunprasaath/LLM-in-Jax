import numpy as np
import jax, jax.numpy as jnp, flax.nnx as nnx
from tokenizers import Tokenizer

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse


@nnx.jit
def _logprobs(model, x):  # x: [1, maxlen] -> [maxlen, vocab]
    return jax.nn.log_softmax(model(x).astype(jnp.float32), axis=-1)[0]


class _NoCache:
    """Lighteval insists on a cache, but its keys ignore the weights, so reuse would return stale scores."""
    def _init_registry(self, registry): pass


class JaxEvalModel(LightevalModel):
    """Lighteval adapter around a (live) GPT; the GPT itself stays a plain nnx.Module."""
    def __init__(self, model, tokenizer_path, maxlen, name="jax_gpt"):
        super().__init__()
        self.model, self.maxlen = model, maxlen
        self.tok = Tokenizer.from_file(str(tokenizer_path))  # own copy: training's tokenizer has padding/truncation on
        self.bos, self.eos = self.tok.token_to_id("<|startoftext|>"), self.tok.token_to_id("<|endoftext|>")
        self.config, self._cache = ModelConfig(model_name=name), _NoCache()

    tokenizer = property(lambda self: self.tok)
    add_special_tokens = property(lambda self: False)
    max_length = property(lambda self: self.maxlen)

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def logprobs(self, ids):  # -> [len(ids), vocab]; padded to maxlen so jit compiles once
        x = np.zeros((1, self.maxlen), np.int32)
        x[0, :len(ids)] = ids
        return _logprobs(self.model, jnp.asarray(x))[:len(ids)]

    def score(self, ctx, cont):  # (sum log p(cont | ctx), is cont the greedy output)
        lp = self.logprobs((ctx + cont)[-self.maxlen:])[-len(cont) - 1:-1]
        cont = jnp.array(cont)
        return float(lp[jnp.arange(len(cont)), cont].sum()), bool((lp.argmax(-1) == cont).all())

    def loglikelihood(self, docs):
        out = []
        for doc in docs:
            ctx, conts = [self.bos] + self.enc(doc.query), [self.enc(c) for c in doc.choices]
            lps, greedy = zip(*(self.score(ctx, c) for c in conts))
            out.append(ModelResponse(logprobs=list(lps), argmax_logits_eq_gold=list(greedy), output_tokens=conts))
        return out

    def loglikelihood_rolling(self, docs):
        out = []
        for doc in docs:
            ids, n = self.enc(doc.query), self.maxlen - 1
            out.append(ModelResponse(logprobs=[sum(self.score([self.bos], ids[i:i + n])[0] for i in range(0, len(ids), n))]))
        return out

    def greedy_until(self, docs):
        out = []
        for doc in docs:
            ids, gen, stops = [self.bos] + self.enc(doc.query), [], doc.stop_sequences or []
            while len(gen) < (doc.generation_size or 64) and len(ids) < self.maxlen:
                nxt = int(self.logprobs(ids)[-1].argmax())
                if nxt == self.eos:
                    break
                ids.append(nxt); gen.append(nxt)
                if any(s in self.tok.decode(gen) for s in stops):
                    break
            text = self.tok.decode(gen)
            for s in stops:
                text = text.split(s)[0]
            out.append(ModelResponse(text=[text], output_tokens=[gen]))
        return out
