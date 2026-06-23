document.getElementById('dataForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const inputData = document.getElementById('inputData').value;
  
    try {
      const response = await fetch('http://127.0.0.1:5000/test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: inputData }),
      });
  
      const result = await response.json();
      document.getElementById('response').textContent = `Response: ${result.message}`;
    } catch (error) {
      document.getElementById('response').textContent = 'Error connecting to backend.';
    }
  });