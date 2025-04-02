const site = "http://localhost:5000"
function showWarning(message) {
    const errorDiv = document.getElementById("error-message");
    if (errorDiv) {
        errorDiv.innerText = message;
    }
}
async function SignUp(event) {
    event.preventDefault();
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    // Validate the email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        showWarning(MESSAGES.warning_invalid_email);
        return;
    }

    // Hash the password using SHA-256

    async function hashPassword(password) {
        const encoder = new TextEncoder();
        const data = encoder.encode(password);
        const hashBuffer = await crypto.subtle.digest("SHA-256", data);
        return Array.from(new Uint8Array(hashBuffer))
            .map(b => b.toString(16).padStart(2, "0"))
            .join("");
    }

    const hashedPassword = await hashPassword(password);

    const requestData = {
        email: email,
        password: hashedPassword
    };

    // Send data to backend
    try {
        const response = await fetch(`${site}/api/signup`, { // Change https to http
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData),
            credentials: "include"
        });

        if (response.ok) {
            // sessionStorage.setItem("isLoggedIn", "true"); don't need
            showWarning(MESSAGES.warning_login_success);
            window.location.href = "/views/transcribe.html";
        } else {
            switch (response.status) {
                case 400:
                    showWarning(MESSAGES.warning_400);
                    break;
                case 404:
                    showWarning(MESSAGES.warning_user_not_found);
                    break;
                case 409:
                    showWarning(MESSAGES.warning_user_already_exists);
                    break;
                case 500:
                    showWarning(MESSAGES.warning_500);
                    break;
                default:
                    showWarning(MESSAGES.warning_generic);
            }
        }
    } catch (error) {
        showWarning(MESSAGES.warning_generic);
    }
}