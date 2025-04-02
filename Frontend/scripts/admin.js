console.log('testing0')
const site = "http://localhost:5000"
document.addEventListener("DOMContentLoaded", async () => {
    try {
        // Check if the user is an admin
        const userResponse = await fetch(`${site}/api/user`, { credentials: "include" });
        const userData = await userResponse.json();
        console.log("Full userData:", userData);
        console.log(userData.user.isAdmin)
        console.log(userData.success)



        if (!userData || !userData.user || userData.user.isAdmin == "false" || userData.user.isAdmin == "undefined") {
            console.log('ohio asl wth')
            alert("Error 403: You do not have permission to view this page.");
            window.location.href = "./home.html"; // Redirect non-admins
            return;
        }
        console.log('ohio 2')
        // Fetch list of users
        const usersResponse = await fetch(`${site}/api/admin/users`, { credentials: "include" });
        const usersData = await usersResponse.json();

        if (usersData.success) {
            displayUsers(usersData.users, usersData.apis);
        } else {
            alert("Failed to fetch users.");
        }
    } catch (error) {
        console.error("Error:", error);
    }
});

// Function to display users in the UI
function displayUsers(users, apis) {
    const usersList = document.getElementById("usersList"); // Assume there's a div with this ID
    usersList.innerHTML = ""; // Clear existing content

    users.forEach(user => {
        const userItem = document.createElement("div");
        userItem.innerHTML = `
            <p>Email: ${user.email} - Admin: ${user.isAdmin}</p>
            <button onclick="deleteUser('${user._id}')">Delete</button>
        `;
        usersList.appendChild(userItem);
    });

    const apiList = document.getElementById("apiList"); // Assume there's a div with this ID
    apis.forEach(api => {
        const apiItem = document.createElement("div");
        apiItem.innerHTML = `
            <p>API: ${api.api} - Method: ${api.Method} - Count: ${api.count}</p>
        `;
        apiList.appendChild(apiItem);
    });
}

// Function to delete a user
async function deleteUser(userId) {
    try {
        const response = await fetch(`${site}/api/admin/delete/${userId}`, {
            method: "DELETE",
            credentials: "include"
        });

        const data = await response.json();
        if (data.success) {
            alert("User deleted successfully.");
            location.reload(); // Refresh the list
        } else {
            alert("Failed to delete user.");
        }
    } catch (error) {
        console.error("Error:", error);
    }
}
