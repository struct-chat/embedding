# E5 Embedding Server

Server to generate embeddings for arbitrary-length documents using E5 model.

Contains a Dockerfile which downloads `e5-small-v2` during build time (hence
avoiding any runtime surprises), and runs app.py.

app.py uses this model to generate embeddings, even for longer documents, by
chunking them up at sentence level, and taking a mean at the end. The sentences
are chunked at 500 tokens each, to avoid crossing the 512 token limit enforced
by E5 models.

## Why build this?

I wanted a Docker based server that I could just run to generate embeddings for
Struct threads. But, I couldn't find it online. Particularly when you want to
tackle longer documents which don't fit the 512 token limit. In those cases, the
document needs to be chunked, ideally at the sentence endings. I couldn't find
this setup available online, so put this together.

I was looking for a Docker-based server, which can expose a simple endpoint to
generate embeddings for documents. The solution needs to deal with lengthy
documents that exceed the 512-token limit enforced by E5 models. Such documents
require intelligent chunking, ideally at sentence boundaries, followed by taking
a mean of the vectors, to work effectively. Since I couldn't find a solution
that met these criteria, I decided to create this setup myself.

## Build

```bash
$ docker build -t <tag> .
$ docker run -p 10002:10002 <tag>
```

By default, this is using `e5-small-v2` model. If you wish to change the model,
you'd have to update the name in both Dockerfile and app.py.

## Usage

When passing a document, use the prefix: `passage: `. When querying for similar
documents, use the prefix `query: `.

If a single sentence exceeds the 512 tokens limit, then we ignore that sentence,
considering it invalid, and adding its tokens to the `invalid_tokens` count.

`num_tokens` is the total number of tokens the document has, including the
`invalid_tokens`.

```
$ curl http://localhost:10002/embed -XPOST -H "content-type: application/json" -d '{"prefix": "passage: ", "text": "This is a sentence."}' | jq

{
  "embedding": [
    -0.07240722328424454,
    0.03763573244214058,
    0.02595547027885914,
    -0.017270879819989204,
    -0.023462336510419846,
    0.05062587931752205,
    0.07775235176086426,
    -0.06687597930431366,
    ...
        -0.06454016268253326,
    0.006377237383276224,
    -0.0546715222299099,
    -0.002629948779940605,
    -0.0011916998773813248,
    0.026298822835087776
  ],
  "invalid_tokens": 0,
  "num_tokens": 5
}
```
