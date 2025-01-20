document.addEventListener("DOMContentLoaded", () => {
    // Ensure active tab toggling
    const tabButtons = document.querySelectorAll(".tab-button");
    tabButtons.forEach(button => {
        button.addEventListener("click", () => {
            tabButtons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
        });
    });

   
    // File upload functionality
    const fileInput = document.getElementById("file-input");
    const uploadButton = document.querySelector(".upload-button");

    uploadButton.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (file) alert(`Selected file: ${file.name}`);
    });
});

document.getElementById("file-input").addEventListener("change", function(event) {
    const formData = new FormData();
    formData.append("file", event.target.files[0]);

    fetch("/upload", {
        method: "POST",
        body: formData
    }).then(response => response.json())
      .then(data => {
          if (data.message === "File uploaded successfully") {
              console.log("File uploaded:", data.file_path);
              detectImage(data.file_path);  // Send file path for detection
          }
      });
});

document.getElementById('file-input').addEventListener('change', function() {
    const fileName = this.files[0] ? this.files[0].name : 'No file chosen';
    document.querySelector('.upload-text').textContent = fileName;
});

// Function to send the uploaded image for detection
function detectImage(filePath) {
    const formData = new FormData();
    formData.append("file_path", filePath);

    fetch("/detect", {
        method: "POST",
        body: formData
    }).then(response => response.json())
      .then(data => {
          if (data.message === "Detection successful") {
              const imagePath = data.image_path;
              console.log("Detection successful, image path:", imagePath);
              const detectedImage = document.getElementById("decimg");
              const detectedSection = document.getElementById("detected-section");
              const downloadBtn = document.getElementById("download-btn");

              detectedImage.src = `/detected_file/${imagePath.split('/').pop()}`;
              detectedSection.style.display = "block";
              downloadBtn.style.display = "inline-block";

              downloadBtn.addEventListener("click", () => {
                  const link = document.createElement("a");
                  link.href = detectedImage.src;
                  link.download = imagePath.split('/').pop();
                  link.click();
              });
          }
      });
}

document.addEventListener('DOMContentLoaded', function() {
    const downloadButton = document.getElementById('download-btn');
    const backButton = document.getElementById('back-btn');
    const detectedImage = document.getElementById('decimg');
    const detectedSection = document.getElementById('output-section');

    
    detectedSection.style.display = 'none';

    // Show the download button when the image is loaded
    detectedImage.onload = function() {
        
        detectedSection.style.display = 'block'; // Show the detected section
    };

    backButton.onload = function() {
        detectedSection.style.display = 'none';
    }

    // Set up the download functionality
    downloadButton.addEventListener('click', function() {
        const imageUrl = detectedImage.src; // Get the image URL
        const link = document.createElement('a'); // Create a temporary link element
        link.href = imageUrl; // Set the link's href to the image URL
        link.download = 'detected_image.jpg'; // Set the download attribute with a filename
        document.body.appendChild(link); // Append the link to the body
        link.click(); // Programmatically click the link to trigger the download
        document.body.removeChild(link); // Remove the link from the document
    });

    // Handle file input click
    document.getElementById('upload-btn').addEventListener('click', function() {
        document.getElementById('file-input').click();
    });

    // Handle file input change
    document.getElementById('file-input').addEventListener('change', function() {
        const form = document.getElementById('upload-form');
        const formData = new FormData(form);

        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.image_path) {
                detectedImage.src = data.image_path; // Set the detected image source
                detectedSection.style.display = 'block'; // Show the detected section
                downloadButton.style.display = 'block'; // Show the download button
            } else {
                alert(data.error || 'An error occurred during detection.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while uploading the file.');
        });
    });
});
