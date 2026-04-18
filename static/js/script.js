document.addEventListener("DOMContentLoaded", function() {
    // Fetch job titles
    fetch('/get-job-categories')
        .then(res => res.json())
        .then(data => {
            const jobSelect = document.getElementById('jobTitle');
            jobSelect.innerHTML = '<option value="">Select a job title</option>';
            data.categories.forEach(job => {
                const option = document.createElement('option');
                option.value = job;
                option.textContent = job;
                jobSelect.appendChild(option);
            });
        });

    const form = document.getElementById('resumeForm');
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const loader = document.querySelector('.loader');
        loader.style.display = 'block';
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'none';
        const formData = new FormData(form);

        fetch('/upload_resume', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => {
                loader.style.display = 'none';
                if (data.error) { alert(data.error); return; }

                // Score progress bar
                const score = Math.min(Math.max(data.score * 100, 0), 100);
                document.getElementById('score').textContent = score.toFixed(0) + '%';

                // Suggestions
                const suggestionsList = document.getElementById('suggestions');
                suggestionsList.innerHTML = '';
                if (Array.isArray(data.suggestions)) {
                    data.suggestions.forEach(s => {
                        const li = document.createElement('li');
                        li.textContent = s;
                        suggestionsList.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.textContent = data.suggestions;
                    suggestionsList.appendChild(li);
                }

                resultDiv.style.display = 'block';
            })
            .catch(err => {
                loader.style.display = 'none';
                alert('Error: ' + err);
            });
    });
});
