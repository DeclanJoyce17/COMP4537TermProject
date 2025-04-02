const site = "https://comp4537termproject-1.onrender.com"

function showWarning(message) {
    const errorDiv = document.getElementById("error-message");
    if (errorDiv) {
        errorDiv.innerText = message;
    }
}

async function login(event) {
    event.preventDefault(); // Prevent form from submitting

    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    // Hash the password using the existing hashPassword function
    const hashedPassword = await hashPassword(password);

    const requestData = {
        email: email,
        password: hashedPassword,
    };

    try {
        // Send POST request to the backend
        const response = await fetch(`${site}/api/login`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestData),
            credentials: "include"
        });

        if (response.ok) {
            sessionStorage.setItem("isLoggedIn", "true");
            const temp = await response.json();
            if (temp.admin === "True") {
                window.location.href = "/views/admin.html";
            } else {
                window.location.href = "/views/transcribe.html";
            }


            // If login is successful, send a GET request to retrieve user info
            const userInfo = await fetch(`${site}/api/user`, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                },
                credentials: "include"
            });
            const data = await userInfo.json();
            if (data.success) {
                showWarning(MESSAGES.warning_login_success + data.user.email);
            }
        } else {
            switch (response.status) {
                case 400:
                    showWarning(MESSAGES.warning_400);
                    break;
                case 403:
                    showWarning(MESSAGES.warning_403);
                    break;
                case 404:
                    showWarning(MESSAGES.warning_404);
                    break;
                case 500:
                    showWarning(MESSAGES.warning_500);
                    break;
                default:
                    showWarning(MESSAGES.warning_login);
            }
        }
    } catch (error) {
        showWarning(MESSAGES.warning_generic);
    }
}

// Hashing function you already have
async function hashPassword(password) {
    const encoder = new TextEncoder();
    const data = encoder.encode(password);
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);
    return Array.from(new Uint8Array(hashBuffer))
        .map(b => b.toString(16).padStart(2, "0"))
        .join("");
}
