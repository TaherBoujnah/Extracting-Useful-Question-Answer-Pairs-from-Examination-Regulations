import { appendMessage, drawFeedbackButtons, drawUserInfo, discloseLimitations } from './ui.js';


let processQueryCache = null;

export function handleWorker(events){

    const { act, results, embeddingCache: updatedCache, tokenEmbeddingCache: updatedTokenCache, query } = events.data;
      
      switch (act) {
        case 'initialized':
          localStorage.setItem('largeEmbedding', JSON.stringify(updatedCache));
          /** use only to download the new embeddings */
          //exportAsJson(JSON.parse(localStorage.getItem("largeEmbedding")), "largeEmbeddings");
          break;
          
        case 'tokeninit':
          localStorage.setItem('smallEmbedding', JSON.stringify(updatedTokenCache));
          /** use only to download the new embeddings */
          //exportAsJson(JSON.parse(localStorage.getItem("smallEmbedding")), "smallEmbedding");
          break;
    
        case 'searchResults':

          const bestTokens = results.map(result => result.bestToken);
          const questions = results.map(result => result.question);
          const bestAnswer = {};

          for (let i = 0; i < questions.length; i++) {
            bestAnswer[questions[i]] = bestTokens[i];
          }
    
          
          drawUserInfo();
          
          /** pass our three results to ui so they can be displayed and worked with in chat */
          results.forEach(result => {
            appendMessage(
              result.question,
              'bot-message',
              result.buttonId,
              result.akkordeonId,
              result.bestToken,
              query,
              result.answer
            );
          });
          drawFeedbackButtons(query, bestAnswer);
          /**discloseLimitations();*/

          break;
    
        case 'topResults':
          /** show results live in result */
          if (processQueryCache) {
            const outputs = results.slice(0, 3).map(result => ({
              question: result.question,
              bestToken: result.bestToken,
              buttonId: result.buttonId,
              akkordeonId: result.akkordeonId,
            }));

            processQueryCache(outputs);
            processQueryCache = null;
          }
          break;
      }
}


export function processQuery(query, worker, cache) {
  processQueryCache = cache;
  worker.postMessage({
    act: 'resultsSemantic',
    query: query,
  });
}


/** export as json; not needed in production */
function exportAsJson(data, fileName) {

  const jsonString = JSON.stringify(data, null, 2); 
  const blob = new Blob([jsonString], { type: 'application/json' }); 
  const url = URL.createbestAnswerectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  
  link.click();
  URL.revokebestAnswerectURL(url);
}