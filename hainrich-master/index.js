import { sendMessage } from './modules/ui.js';
import { tokenizeText, initializeModel } from './modules/semanticSearch.js';
import { extractFAQfromHTML } from './modules/faqData.js';
import { loadEmbedding, removeOutdatedEntries } from './modules/embeddings.js'
import { handleWorker, processQuery } from './modules/workerInterface.js'


const worker = new Worker('./worker.js', { type: 'module' });

/** get the up to date FAQ from site */
const textData = await extractFAQfromHTML("FAQ.html");

/** import precompiled embeddings and check for missing entries */
console.time("loadLargeEmbeddings");
let embeddingStack = await loadEmbedding('largeEmbedding', './embeddings/largeEmbedding.json');
console.timeEnd("loadLargeEmbeddings");

console.time("loadSamllEmbeddings");
let tokenEmbeddingStack = await loadEmbedding('smallEmbedding', './embeddings/smallEmbedding.json');
console.timeEnd("loadSamllEmbeddings");

/** throw out embeddings for entries not present on site any more */
const checkFAQ = new Set(textData.map(faq => faq.qPlusA));
removeOutdatedEntries(embeddingStack, checkFAQ);

const checkTokensAnswer = new Set();
textData.forEach(faq => tokenizeText(faq.answer).forEach(token => checkTokensAnswer.add(token)));
removeOutdatedEntries(tokenEmbeddingStack, checkTokensAnswer);

/** load up to date embeddings into local browser storage */
localStorage.setItem('largeEmbedding', JSON.stringify(embeddingStack));
localStorage.setItem('smallEmbedding', JSON.stringify(tokenEmbeddingStack));


const inputElement = document.getElementById('user-input');
const resultsContainer = document.getElementById('results');


/** letting users know the site is not broken by showing them a loading animation until model is loaded */
let dotCount = 0;
let loadingAnimation = null;
  
function animateDots() {
  dotCount = (dotCount + 1) % 4;
  let dots = '.'.repeat(dotCount);
  inputElement.placeholder = `Loading ${dots}`;
  loadingAnimation = setTimeout(animateDots, 300);
}

animateDots();

let topResult = null;
inputElement.addEventListener('input', function (event) {
  const query = event.target.value.trim();
  
  /** hide results when no query is typed */
  if (!query) {
    resultsContainer.innerHTML = '';
    resultsContainer.style.display = 'none';
    return;
  }
  
  resultsContainer.style.display = 'block';
  resultsContainer.innerHTML = '';
  
  processQuery(query, worker, function (outputs) {
    window.focus();
    outputs.forEach(async output => {
      
      /** display live results */
      const resultLink = document.createElement('div');
      resultLink.className = 'result';
      
      /** display question text */
      const questionDiv = document.createElement('div');
      questionDiv.className = 'result-question';
      questionDiv.innerHTML = `<strong>${output.question}</strong>`;
      resultLink.appendChild(questionDiv);
      
      /** display our most relevant token to question */
      const bestTokenDiv = document.createElement('div');
      bestTokenDiv.className = 'result-token';
      let tokenText = output.bestToken;
      if (tokenText.length > 100) {
        tokenText = tokenText.substring(0, 100);
      }
      tokenText = '> ' + tokenText + ' ...';
      bestTokenDiv.textContent = tokenText;
      resultLink.appendChild(bestTokenDiv);
      
      window.focus();
      /** open the answer field in chat view for selected panel */
      resultLink.onclick = function () {
        linkAndClick(output.question);
      };
      
      resultsContainer.appendChild(resultLink);
    });
    
    if (outputs.length > 0) {
      topResult = outputs[0];
    }
  });
});

/** search for query on enter press and put it in chat */
document.getElementById('user-input').addEventListener('keypress', function (event) {
  if (event.key === 'Enter') {
    event.preventDefault();
    if (topResult) {
      linkAndClick(topResult.question);
    }
  }
});

export function linkAndClick(output) {
  saveMessage();
  setTimeout(() => {
    const encodedQuestion = encodeURIComponent(output);
    const panels = document.querySelectorAll(
      `.info-panel[data-question="${encodedQuestion}"]`
    );
    if (panels.length > 0) {
      panels[panels.length - 1].click();
    }
  }, 250);
}

/** once model is initialized loading message is removed and user can start typing */
worker.addEventListener( 'message',function(events) {
  if (events.data.type === 'ready') {
    readyForInput();
    console.log("Ready");
  }
});


/** show results in chat window if "ask" is clicked */
function saveMessage(){
  return sendMessage(worker);
}

function sendAsk(){
  if (topResult) {
    linkAndClick(topResult.question);
  }
}

function readyForInput(){
  clearTimeout(loadingAnimation);
  inputElement.disabled = false;
  inputElement.placeholder = "Try asking: 'minimum admission grade' or 'I'm from outside the EU'";
  inputElement.focus();
}


worker.onmessage = (events) => handleWorker(events);
window.sendAsk = sendAsk;

/** give worker work */
worker.postMessage({ act: 'initialize', textData, embeddingStack, tokenEmbeddingStack });








