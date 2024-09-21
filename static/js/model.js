function UPLOAD(){
    

    const fileInput = document.getElementById("file-input");
    if (fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);


    fetch('/upload_file', {
                method: 'POST',
                body: formData
            })
    .then(response => response.json())
    .catch(error => {
        console.error('Error:', error);  // Handle errors
    });

    document.getElementById('uploadform').submit();

    var x = document.getElementsByTagName('main')[0];
    var i = document.getElementById('uploadform');
    var y = document.createElement('div');
    var z = document.createElement('H1');
    z.innerText = "UPLOADING....";

    i.parentElement.removeChild(i);
    y.classList.add('containermain');
    y.appendChild(z);
    x.appendChild(y);
  }