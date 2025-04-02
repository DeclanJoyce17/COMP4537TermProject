const site = "http://localhost:5000"

async function forgotPassword(event) {
    event.preventDefault(); // Prevent form from submitting
    const email = document.getElementById("email").value;

    // Validate the email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        alert("Please enter a valid email address.");
        return;
    }
    const requestData = {
        email: email
    };

    try {
        // Send POST request to the backend
        const response = await fetch(`${site}/api/reset`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestData),
        });

        if (response.ok) {
            document.querySelector("#confirmation").innerHTML = "Password Reset Email is Sent"

        } else {
            // Handle login failure
            alert("Email failed to send. Please check your credentials and try again.");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred. Please try again later.");
    }
}

async function resetPassword(event) {
    event.preventDefault(); // Prevent form from submitting

    const urlParams = new URLSearchParams(window.location.search);
    const email = urlParams.get('email');
    const token = urlParams.get('token');

    const password = document.getElementById("password").value;
    const passwordConfirm = document.getElementById("passwordConfirm").value;
    // ensures passwords match
    if (password.localeCompare(passwordConfirm) != 0) {
        alert("Passwords don't match");
        return;
    }
    const hashedPassword = await hashPassword(password);

    const requestData = {
        email: email,
        token: token,
        newPassword: hashedPassword
    };

    try {
        // Send POST request to the backend
        const response = await fetch(`${site}/api/reset/resetPassword`, {
            method: "PUT",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestData),
            // credentials: "include"
        });

        if (response.ok) {
            document.querySelector("#confirmation").innerHTML = "Password is reset"

        } else {
            // Handle login failure
            alert("Password reset failed. Please check your credentials and try again.");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred. Please try again later.");
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