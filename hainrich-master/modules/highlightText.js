

/** URL highlights stay in DOM until refresh of page; not ideal */
/** might leave it out */
export function highlightText(textToHighlight) {
  const text = encodeURIComponent(textToHighlight);
  
  let windows = window;
  while (windows !== windows.parent) {
    windows = windows.parent;
  }
  
  const currentUrl = windows.location.origin + windows.location.pathname + windows.location.search;
  const newUrl = currentUrl + '#:~:text=' + text;
  windows.location.href = newUrl;
}




