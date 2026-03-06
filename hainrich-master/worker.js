
import { 
  initializeModel, 
  calcFAQEmbeddings, 
  precalcTokenEmbeddings, 
  semanticSearch, 
  findBestToken,
  getEmbeddingCache,
  getTokenEmbeddingCache,
  setCaches
} from './modules/semanticSearch.js';





let textData = [];
let faqEmbeddings = [];

self.onmessage = async function (event) {
  const { act, query, textData: newtextData, embeddingStack, tokenEmbeddingStack } = event.data;
  
  switch (act) {
    case 'initialize':

      console.time("Model Loading Time");
      await initializeModel();
      console.timeEnd("Model Loading Time");

      console.time("calcEmbeddings");
      
      textData = newtextData;
      setCaches(embeddingStack, tokenEmbeddingStack);
      
      faqEmbeddings = await calcFAQEmbeddings(textData, "qPlusA");
      
      await precalcTokenEmbeddings(textData);
      console.timeEnd("calcEmbeddings");
      postMessage({ act: 'initialized', embeddingCache: getEmbeddingCache() });
      postMessage({ act: 'tokeninit', tokenEmbeddingCache: getTokenEmbeddingCache() });

      self.postMessage({
        type: 'ready'
      });
      break;
    
    case 'semanticSearch':
      
      const faqResults = await semanticSearch(query, textData, faqEmbeddings);
      const resultsTokenChat = await getResultsWithTokens(faqResults, query);
      postMessage({ act: 'searchResults', results: resultsTokenChat, query });
      break;

    case 'resultsSemantic':

      const faqResult = await semanticSearch(query, textData, faqEmbeddings);
      const resultsTokenResults = await getResultsWithTokens(faqResult, query);
      postMessage({ act: 'topResults', results: resultsTokenResults, query });
      break;
        
  }
};

/** extract the best token in side the answer of a panel for a given query */
async function getResultsWithTokens(faqResults, query) {
  return await Promise.all(
    faqResults.map(async (result) => {
      const { bestToken } = await findBestToken(query, result.answer);
      return { ...result, bestToken};
    })
  );
}
