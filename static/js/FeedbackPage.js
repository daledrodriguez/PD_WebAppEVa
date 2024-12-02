document.addEventListener("DOMContentLoaded", () => {
  const feedbackForm = document.getElementById("feedback-form");
  const feedbackContent = document.getElementById("feedback-content");
  let stopTypingLoop = false; // To control the loading messages loop

  // Initialize with the default "Feedback will appear here once generated..."
  feedbackContent.innerHTML = "<p>Feedback will appear here once generated...</p>";

  feedbackForm.addEventListener("submit", async (e) => {
    e.preventDefault(); // Prevent the default form submission

    // Clear any previous feedback and start showing loading messages
    feedbackContent.innerHTML = "";
    stopTypingLoop = false;
    typeMessagesLoop(feedbackContent, [
      "Generating Feedback...",
      "Eva is Analyzing your code for you...",
    ]);

    const formData = new FormData(feedbackForm);

    try {
      // Send POST request to the backend
      const response = await fetch("/feedback", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        stopTypingLoop = true; // Stop loading messages
        feedbackContent.innerHTML = ""; // Clear any loading messages

        // Display the feedback dynamically with syntax highlighting
        displayFeedback(feedbackContent, data.feedback);
      } else {
        stopTypingLoop = true; // Stop loading messages
        feedbackContent.innerHTML = `<p class="error">${data.error || "An error occurred while generating feedback."}</p>`;
      }
    } catch (error) {
      stopTypingLoop = true; // Stop loading messages
      feedbackContent.innerHTML = `<p class="error">Failed to connect to the server.</p>`;
    }
  });

  // Function to continuously loop typing messages
  async function typeMessagesLoop(element, messages) {
    while (!stopTypingLoop) { // Loop as long as stopTypingLoop is false
      for (const message of messages) {
        if (stopTypingLoop) return; // Stop if feedback is ready
        element.textContent = ""; // Clear content for the next message
        await typeText(element, message, 50); // Type message dynamically
        await new Promise((resolve) => setTimeout(resolve, 1000)); // Pause before the next message
      }
    }
  }

  // Function to type text dynamically
  function typeText(element, text, speed = 20) {
    return new Promise((resolve) => {
      let index = 0;

      function type() {
        if (index < text.length) {
          element.textContent += text[index];
          index++;
          setTimeout(type, speed);
        } else {
          resolve(); // Resolve the promise when typing is complete
        }
      }

      type();
    });
  }

  async function displayFeedback(element, feedback) {
    const sections = feedback.split(/(Strengths:|Weaknesses:|Suggested steps:|Code implementation:|Explanation:|Conclusion:|Suggestive steps and implementation:)/g);
    element.innerHTML = ""; // Clear existing content

    for (let i = 0; i < sections.length; i++) {
        const section = sections[i].trim();
        if (!section) continue;

        // Check for section titles and format them
        if (["Strengths:", "Weaknesses:", "Suggested Steps:", "Code implementation:", "Explanation:", "Conclusion:", "Suggestive steps and implementation"].includes(section)) {
            const title = document.createElement("div");
            title.className = "section-title";
            title.textContent = section;
            element.appendChild(title);
        } else {
            // Process text and code separately
            const codeBlockRegex = /```([a-z]*)\n([\s\S]*?)```/g;
            let lastIndex = 0;

            let match;
            while ((match = codeBlockRegex.exec(section)) !== null) {
                const nonCodeText = section.slice(lastIndex, match.index).trim();
                if (nonCodeText) {
                    const paragraphs = nonCodeText.split("\n").filter((line) => line.trim());
                    for (const paragraph of paragraphs) {
                        const p = document.createElement("p");
                        element.appendChild(p);
                        await typeText(p, paragraph, 20); // Adjust typing speed here
                    }
                }

                // Add code block
                const pre = document.createElement("pre");
                const codeElement = document.createElement("code");
                codeElement.className = `language-${match[1] || "plaintext"}`;
                element.appendChild(pre);
                pre.appendChild(codeElement);

                await typeText(codeElement, match[2].trim(), 5);
                Prism.highlightElement(codeElement);
                lastIndex = codeBlockRegex.lastIndex;
            }

            // Handle any remaining non-code text
            const remainingText = section.slice(lastIndex).trim();
            if (remainingText) {
                const paragraphs = remainingText.split("\n").filter((line) => line.trim());
                for (const paragraph of paragraphs) {
                    const p = document.createElement("p");
                    element.appendChild(p);
                    await typeText(p, paragraph, 20);
                }
            }
        }

        // Scroll to bottom
        element.scrollTop = element.scrollHeight;
    }
}


});
