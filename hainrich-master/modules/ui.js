import { submitFeedback } from "./feedback.js";

/** handle what needs to be displayed in chat container */
export function appendMessage(content, className, linkId = null, akkordeonId = null, bestToken = null, query = null, answer = null) {
  const messageContainer = document.createElement('div');
  
  if (className === 'user-message') {
    const messageDiv = document.createElement('div');
    messageDiv.textContent = content;
    messageDiv.className = className;
    messageContainer.appendChild(messageDiv);

  } else {
    
    messageContainer.className = 'bot-message-container';

    const numberBox = document.createElement('div');
    numberBox.className = 'number-box';
    
    const messageDiv = document.createElement('div');
    messageDiv.textContent = content;
    messageDiv.className = className;
    
    const panelDiv = document.createElement('div');
    panelDiv.innerHTML = '<i class="fa-solid fa-down-long"></i>';
    panelDiv.className = 'info-panel';
    panelDiv.setAttribute('data-question', encodeURIComponent(content));
    
    messageDiv.onclick = function(event) {
      event.stopPropagation();
      panelDiv.click();
    }
    
    panelDiv.onclick = function (event) {
      event.stopPropagation();
      let nextElem = messageContainer.nextElementSibling;
      if (nextElem && nextElem.classList.contains('answer-container')) {
        nextElem.remove();
        panelDiv.innerHTML = '<i class="fa-solid fa-down-long"></i>';
        panelDiv.classList.remove('active');
        return;
      }
      panelDiv.innerHTML = '<i class="fa-solid fa-up-long"></i>';
      
      let highlightToken = answer.replace(bestToken, `<span class="highlight">${bestToken}</span>`);
      
      const answerContainer = document.createElement('div');
      answerContainer.className = 'answer-container';
      answerContainer.innerHTML = highlightToken;
      
      messageContainer.parentNode.insertBefore(answerContainer, messageContainer.nextSibling);
      panelDiv.classList.add('active');
    }

    messageContainer.appendChild(numberBox);
    messageContainer.appendChild(messageDiv);
    messageContainer.appendChild(panelDiv);
  }

  const messagesDiv = document.getElementById('messages');
  messagesDiv.appendChild(messageContainer);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

/** gets query from input field and performs search */
export function sendMessage(worker) {
  const userInput = document.getElementById('user-input');
  const message = userInput.value.trim();
  
  
  const resultsContainer = document.getElementById('results');
  
  /** hide results when message is sent */
  resultsContainer.innerHTML = '';
  setTimeout(() => {
    resultsContainer.style.display = 'none';
  }, 500);
  
  resultsContainer.style.display = 'block';
  resultsContainer.innerHTML = '';

  if (message !== "") {
    appendMessage(message, 'user-message');
    userInput.value = '';
    worker.postMessage({
      act: 'semanticSearch',
      query: message,
    });
  }
}

/** new feedback buttons with font-awesome style */
export function drawFeedbackButtons(query, bestTokens) {
  const messagesDiv = document.getElementById('messages');
  const feedbackContainer = document.createElement('div');
  feedbackContainer.className = 'feedback-container';

  const plusButton = document.createElement('button');
  plusButton.innerHTML = '<i class="fa-solid fa-thumbs-up"></i>';
  plusButton.className = 'feedback-button thumbs-up';
  plusButton.onclick = function () {
    submitFeedback(query, bestTokens, 'Yes');
  };

  const minusButton = document.createElement('button');
  minusButton.innerHTML = '<i class="fa-solid fa-thumbs-down"></i>';
  minusButton.className = 'feedback-button thumbs-down';
  minusButton.onclick = function () {
    submitFeedback(query, bestTokens, 'No');
  };

  feedbackContainer.appendChild(plusButton);
  feedbackContainer.appendChild(minusButton);
  messagesDiv.appendChild(feedbackContainer);
}

/** Intro panel for chat responses */
export function drawUserInfo() {
  const messagesDiv = document.getElementById('messages');
  const infoDiv = document.createElement('div');
  infoDiv.className = 'bot-intro';
  infoDiv.innerHTML = "This might be useful:<br>";
  messagesDiv.appendChild(infoDiv);
}


/** new message to inform user of the limited usability; out for now*/
export function discloseLimitations() {

  const messagesDiv = document.getElementById('messages');
  const infoDiv = document.createElement('div');

  infoDiv.className = 'info-for-user';
  infoDiv.innerHTML = "Unfortunately, we're unable to provide direct links to the panel due to technical and policy restrictions.<br><ln><ln>Please copy the relevant question to your clipboard and use your browser's search function to locate the relevant panel on this page.";
  
  messagesDiv.appendChild(infoDiv);
}


