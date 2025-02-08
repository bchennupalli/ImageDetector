document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("imageInput");
    const dropArea = document.getElementById("dropArea");
    const preview = document.getElementById("preview");
    const uploadBtn = document.getElementById("uploadBtn");
    const statusText = document.getElementById("status");
    const progressContainer = document.querySelector(".progress-container");
    const progressBar = document.getElementById("progressBar");

    // Drag & Drop Events
    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("dragover");
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dropArea.classList.remove("dragover");

        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    imageInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        handleFile(file);
    });

    function handleFile(file) {
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = "block";
                uploadBtn.style.display = "inline-block";
            };
            reader.readAsDataURL(file);
        }
    }

    // Upload Image
    uploadBtn.addEventListener("click", function () {
        const file = imageInput.files[0];

        if (!file) {
            statusText.innerHTML = "⚠️ Please select an image first!";
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        // Show Progress Bar
        progressContainer.style.display = "block";
        progressBar.style.width = "0%";

        fetch("https://imagedetector-qrh5.onrender.com/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            progressBar.style.width = "100%"; // Complete progress bar
            setTimeout(() => progressContainer.style.display = "none", 500); // Hide after 0.5s

            if (data.result) {
                statusText.innerHTML = `<strong>Prediction:</strong> ${data.result} <br> <strong>Probability:</strong> ${data.probability}`;
            } else {
                statusText.innerHTML = "⚠️ Error: No result received.";
            }
        })
        .catch(error => {
            statusText.innerHTML = "❌ Error uploading image.";
            console.error("Upload Error:", error);
        });
    });
});
