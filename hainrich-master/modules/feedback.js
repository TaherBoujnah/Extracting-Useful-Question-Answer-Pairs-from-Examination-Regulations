


/** feedback now opens a Google Form prefilled with our parameters */
export function submitFeedback(query, bestAnswers, feedback) {

  const url = 'https://docs.google.com/forms/d/e/1FAIpQLScv6hLxUZN42pZYkW28FL3WkXZbL_dzdGo0sE-LdeJbIzTi2g/viewform?usp=pp_url';

  const bestAnswersAsString = Object.entries(bestAnswers)
    .map(([question, token]) => `> Question: ${question}\n> Answer: ${token}`)
    .join('\n\n');

  const encodedQuery = encodeURIComponent(query);
  const encodedBestAnswers = encodeURIComponent(bestAnswersAsString);

  const formFeedback = `entry.293085603=${feedback}`;
  const formQuery = `entry.430302024=${encodedQuery}`;
  const formAnswer = `entry.1738087812=${encodedBestAnswers}`;

  const newUrl = `${url}&${formFeedback}&${formQuery}&${formAnswer}`;
  window.open(newUrl, '_blank');
}

