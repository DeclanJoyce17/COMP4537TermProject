const express = require('express');
const session = require('express-session');
const mongoose = require('mongoose');
const { User, ResetToken } = require("./models");


const jwt = require('jsonwebtoken');
const cookieParser = require('cookie-parser');
const crypto = require('crypto');

const app = express();
const cors = require('cors');
require('dotenv').config();
const path = require('path');


app.use(express.urlencoded({ extended: true }));
app.use(express.json());

const port = process.env.PORT || 3000;
const mongoUri = process.env.MONGO_URI;
const JWT_SECRET = process.env.JWT_SECRET || crypto.randomBytes(64).toString('hex');

// Routing for API Call
const transcribeRoutes = require('./routes/transcribe');
app.use('/transcribe', transcribeRoutes);
// const loginRoutes = require("./routes/loginRoute");
// app.use('/api/login', loginRoutes);
const forgotRoutes = require("./routes/forgotRoute");
app.use('/api/reset', forgotRoutes);

// Serve static files from the Frontend folder
app.use(express.static(path.join(__dirname, '../Frontend')));

// Route to serve transcribe.html
app.get('/transcribe', (req, res) => {
    res.sendFile(path.join(__dirname, '../Frontend/views/transcribe.html'));
});

app.get('/signup', (req, res) => {
    res.sendFile(path.join(__dirname, '../Frontend/views/signup.html'));
});

app.use(cors({
    origin: "*", // Allow all origins (for development)
    methods: "GET,POST,PUT, DELETE,OPTIONS",
    allowedHeaders: "Content-Type, Authorization",
    credentials: true
}));

async function initMongoDB() {
    try {
        await mongoose.connect(mongoUri, {
            useNewUrlParser: true,
            useUnifiedTopology: true
        });
        console.log('✅ Connected to MongoDB!');
    } catch (err) {
        console.error("❌ Connection error:", err);
        process.exit(1); // Exit if MongoDB connection fails
    }
}

app.use(cookieParser());

//---------- Middlewares ---------------//

//admin middleware
const isAdminMiddleware = (req, res, next) => {
    const token = req.cookies.token;
    if (!token) return res.status(401).json({ success: false, message: "Unauthorized" });

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        if (!decoded.isAdmin) {
            return res.status(403).json({ success: false, message: "Forbidden" });
        }
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(401).json({ success: false, message: "Invalid token" });
    }
};

// Handle signup requests
app.post("/api/signup", async (req, res) => {

    try {
        const { email, password } = req.body;
        const emailSite = email.split("@")[1];
        const isAdmin = (emailSite === "admin.com")
        console.log(emailSite);

        console.log("Received Signup Request:");
        console.log("email:", email);
        console.log("Hashed Password:", password);


        // Check if the user already exists in MongoDB
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: "User already exists." });
        }

        // Create a new user in the database
        const newUser = new User({
            email,
            password,
            isAdmin // Ensure password is hashed before storing in production
        });

        await newUser.save();
        console.log("User signed up successfully:", newUser);
        const token = jwt.sign(
            { email: newUser.email, isAdmin: newUser.isAdmin },
            JWT_SECRET,
            { expiresIn: "1h" }
        );
        // Set JWT as an HTTP-only cookie
        res.cookie("token", token, {
            httpOnly: true,
            secure: false, // Set to true in production if you use HTTPS, will do it later
            // maxAge: 5000 // I was just testing, so I set it to 5 seconds
            maxAge: 60 * 60 * 1000 // 1 hour in ms
        });


        res.status(201).json({ message: "Signup successful!" });

    } catch (error) {
        console.error("Error during signup:", error);
        res.status(500).json({ message: "An error occurred. Please try again." });
    }
});

app.post("/api/login", async (req, res) => {
    const { email, password } = req.body;

    try {
        // Check if user exists
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(401).json({ message: "Invalid credentials" });
        }

        // Compare hashed password from the client with the hashed password in the database
        const hashedPasswordFromDB = user.password;
        const hashedPasswordFromClient = password;

        if (hashedPasswordFromClient !== hashedPasswordFromDB) {
            return res.status(401).json({ message: "Invalid credentials" });
        }
        console.log("Logged In");

        // Generate JWT token
        const token = jwt.sign(
            { email: user.email, isAdmin: user.isAdmin },
            JWT_SECRET,
            { expiresIn: "1h" }
        );
        // set jwt as an http cookie
        res.cookie("token", token, {
            httpOnly: true,
            secure: false, // Change to true when serving over HTTPS in production
            maxAge: 60 * 60 * 1000
            // maxAge: 5000 // I was just testing, so I set it to 5 seconds
        });

        if (user.isAdmin === "true") {
            res.status(200).json({ message: "Login successful", admin: "True" });

        } else {
            res.status(200).json({ message: "Login successful", admin: "False" });
        }
    } catch (error) {
        console.error("Login error:", error);
        res.status(500).json({ message: "An error occurred. Please try again." });
    }
});

app.get("/api/user", (req, res) => {
    const token = req.cookies.token;
    if (!token) {
        return res.json({ success: false, message: "No user logged in" });
    }

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        res.json({
            success: true,
            user: {
                email: decoded.email,
                isAdmin: decoded.isAdmin
            }
        });
    } catch (err) {
        res.json({ success: false, message: "Invalid token" });
    }
});

// Get list of all users (Admin only)
app.get("/api/admin/users", isAdminMiddleware, async (req, res) => {
    try {
        const users = await User.find({}, "email isAdmin"); // Fetch only necessary fields
        res.json({ success: true, users });
    } catch (error) {
        res.status(500).json({ success: false, message: "Error fetching users" });
    }
});

// Delete user (Admin only)
app.delete("/api/admin/delete/:id", isAdminMiddleware, async (req, res) => {
    try {
        await User.findByIdAndDelete(req.params.id);
        res.json({ success: true, message: "User deleted" });
    } catch (error) {
        res.status(500).json({ success: false, message: "Error deleting user" });
    }
});



// Start Express Server AFTER DB Connection
initMongoDB().then(() => {
    // Start the server
    app.listen(port, () => {
        console.log(`🚀 Server running on http://localhost:${port}`);
    });
});


