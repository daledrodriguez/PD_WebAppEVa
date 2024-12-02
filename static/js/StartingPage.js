// JavaScript for Modal
const modal = document.getElementById("OpenModal");
const openModalBtn = document.getElementById("OpenLogInBtn");
const closeModalBtn = document.querySelector(".CloseBtn");
const feedbackURL = document.getElementById("feedback-url").value

//for navigation of feedback page
document.getElementById('BtnSubmit').addEventListener('click', navigateToFeedback);
document.getElementById('BtnSubmitG').addEventListener('click', navigateToFeedback);
function navigateToFeedback() {
  
  document.querySelector('.content-container').classList.add('fade-out');
  
  // Show the loading screen after the content fade-out
  setTimeout(() => {
      document.getElementById('loading-screen').classList.remove('hidden');
  }, 500); // Wait for the fade-out transition

  // Redirect to feedback page after 2 seconds
  setTimeout(() => {
      window.location.href = feedbackURL;
  }, 2000); 
}

// Open Modal
openModalBtn.addEventListener("click", () => {
  // modal.classList.add('show');
  modal.classList.remove("hidden");
});

// Close Modal
closeModalBtn.addEventListener("click", () => {
    modal.classList.add("hidden");
});

// Close Modal When Clicking Outside
window.addEventListener("click", (event) => {
    if (event.target === modal) {
        modal.classList.add("hidden");
    }
});