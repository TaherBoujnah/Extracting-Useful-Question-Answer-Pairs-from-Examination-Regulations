import { pipeline, env } from './transformers.js';


/** set transformers.js variables so that nothing is loaded externally */
env.allowRemoteModels = false; 
env.useBrowserCache = true;
env.localModelPath = './models/';
env.backends.onnx.wasm.wasmPaths = './wasm/'




let embedder;
let embeddingCache = {};
let tokenEmbeddingCache = {};
let queryEmbeddingCache = {};


/** getting extraction model ready */
export async function initializeModel() {

  embedder = await pipeline(
        'feature-extraction', 
        'Xenova/all-MiniLM-L6-v2'
      
  );

  console.log("Model Loaded");

}

/** cosine similarity works out of the box */
/** https://www.restack.io/p/similarity-search-answer-javascript-cosine-similarity-cat-ai */
export function cosineSimilarity(vectorA, vectorB) {

  const dotProduct = vectorA.reduce((sum, value, index) => sum + value * vectorB[index], 0);
  const magnitudeA = Math.sqrt(vectorA.reduce((sum, value) => sum + value * value, 0));
  const magnitudeB = Math.sqrt(vectorB.reduce((sum, value) => sum + value * value, 0));
  return dotProduct / (magnitudeA * magnitudeB);

}



/** various function to calculate the right embeddings */
/** -------------------------------------------------- */
function roundEmbedding(embedding) {
  return embedding.data.map(v => Number(v.toFixed(3)));
}

export async function getAndRoundEmbedding(text) {

  const embedding = await embedder(text, { 
                        pooling: 'mean', 
                        normalize: true 
                    });

  return roundEmbedding(embedding);
}

export async function computeEmbedding(text) {
  return getCachedEmbedding(text, embeddingCache);
}

export async function computeTokenEmbedding(text) {
  return getCachedEmbedding(text, tokenEmbeddingCache);
}

export async function computeQueryEmbedding(text) {
  return getCachedEmbedding(text, queryEmbeddingCache);
}

export async function calcFAQEmbeddings(textData, type) {
  return Promise.all(textData.map(item => computeEmbedding(item[type])));
}

export async function getEmbedding(text) {

  const result = await embedder(text, { 
                    pooling: 'mean',
                    normalize: true 
                  });

  return result.data;
}
/** -------------------------------------------------- */

/** checking if embedding already exists, if not calulate it */
export async function getCachedEmbedding(text, cache) {

  if (cache[text]) {
    return cache[text];
  }
  const emb = await getAndRoundEmbedding(text);
  cache[text] = emb;

  return emb;
}

export function getEmbeddingCache() {
  return embeddingCache;
}

export function getTokenEmbeddingCache() {
  return tokenEmbeddingCache;
}


export function setCaches(newEmbeddingCache, newTokenEmbeddingCache) {

  embeddingCache = newEmbeddingCache;
  tokenEmbeddingCache = newTokenEmbeddingCache;

}


/** turn a given text snippet in to tokens */
/** this could still be optimized with more elaborate regular expessions; good for now */
export function tokenizeText(text) {

  const tokens = [];
  const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.length > 0);
  let buffer = "";
  for (let i = 0; i < sentences.length; i++) {
    buffer = buffer ? buffer + " " + sentences[i] : sentences[i];
    if (buffer.length >= 10 || i === sentences.length - 1) {
      tokens.push(buffer);
      buffer = "";
    }
  }
  return tokens;
}

/** getting token embeddings ready for later use in realtime search */
export async function precalcTokenEmbeddings(textData) {

  const allTokens = textData.flatMap(item => tokenizeText(item.answer));
  const uniqueTokens = Array.from(new Set(allTokens));
  await Promise.all(uniqueTokens.map(token => computeTokenEmbedding(token)));

}


/** give each FAQ entry a score on how similiar it is to the given query */
/** return only the three most similiar */
export async function semanticSearch(query, textData, faqEmbeddings, topK = 3) {
  
  const queryEmbedding = await computeQueryEmbedding(query);

  return faqEmbeddings
    .map((emb, index) => ({
      question: textData[index].question,
      answer: textData[index].answer,
      buttonId: textData[index].buttonId,
      akkordeonId: textData[index].akkordeonId,
      qPlusA: textData[index].qPlusA,
      token: textData[index].token,
      score: cosineSimilarity(queryEmbedding, emb),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}


export async function findBestToken(query, text) {
  
  const tokens = tokenizeText(text);
  const queryEmbedding = await computeQueryEmbedding(query);
  let bestToken = null;
  let bestScore = -Infinity;

  for (const token of tokens) {
    const tokenEmb = await computeTokenEmbedding(token);
    const score = cosineSimilarity(queryEmbedding, tokenEmb);
    if (score > bestScore) {
      bestScore = score;
      bestToken = token;
    }
  }
  return { bestToken };
}

