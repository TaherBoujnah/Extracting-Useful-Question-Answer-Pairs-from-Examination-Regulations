import { convertToFloat32 } from './faqData.js';


/** import embeddings from JSON files in file system */
async function loadJsonData(filePath) {
  try {
    const response = await fetch(filePath);
    if (!response.ok) return null;
    const jsonData = JSON.parse(await response.text());
    return convertToFloat32(jsonData);
  } catch (err) {
    return null;
  }
}

/** check if embeddings are presenet in local storage. If not, try getting them from JSON file */
export async function loadEmbedding(key, filePath) {
  const retrievedEmb = JSON.parse(localStorage.getItem(key));
  if (retrievedEmb && Object.keys(retrievedEmb).length > 0) {

    return convertToFloat32(retrievedEmb);
  } else {
    return loadJsonData(filePath) || {};
  }
}

/** get rid of old embeddings */
export function removeOutdatedEntries(stack, validKeys) {
  Object.keys(stack).forEach(key => {
    if (!validKeys.has(key)) {
      delete stack[key];
    }
  });
}
