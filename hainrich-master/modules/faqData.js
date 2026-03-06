import { tokenizeText } from './semanticSearch.js';


/** import the questions, answers and panel Ids from HHU FAQ site */
/** panel Ids needed for locating and highlighting chat response */
export async function extractFAQfromHTML(url) {
  
  const response = await fetch(url);
  const htmlText = await response.text();
  const parser = new DOMParser();
  const doc = parser.parseFromString(htmlText, "text/html");

  const result = [];
  const akkordeonGroup = doc.querySelectorAll('.panel-group');

  akkordeonGroup.forEach(group => {
    const akkordeonId = group.id;
    const panels = group.querySelectorAll('.panel');
    panels.forEach(panel => {
      const questionButton = panel.querySelector('button');
      const panelBody = panel.querySelector('.panel-body');
      if (questionButton && panelBody) {
        const buttonId = questionButton.id;
        /** extracting the relevant parts */
        const questionText = questionButton.querySelector('.text-box')?.textContent.trim();
        const answerText = panelBody.querySelector('.ce-bodytext')?.textContent.trim() || "";

        result.push({
          akkordeonId,
          buttonId,
          question: questionText,
          answer: answerText,
          qPlusA: questionText + " " + answerText,
          tokens: tokenizeText(answerText)
        });
      }
    });
  });
  return result;
}

/** conversion is needed for storage of embeddings */
export function convertToFloat32(inputEmbeddings) {
  const convertedEmbeddings = {};
  for (const [key, value] of Object.entries(inputEmbeddings)) {
    convertedEmbeddings[key] = new Float32Array(Object.values(value));
  }
  return convertedEmbeddings;
}