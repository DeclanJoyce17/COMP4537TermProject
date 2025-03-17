const express = require('express');
const session = require('express-session');
const mongoose = require('mongoose');
const app = express();
require('dotenv').config();
const path = require('path');

// Serve static files
app.use(express.static(path.join(__dirname, '../Frontend/views'))); // Serve HTML files
app.use('/scripts', express.static(path.join(__dirname, '../Frontend/scripts'))); // Serve JavaScript files

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

const port = process.env.PORT || 3000;
const mongoUri = process.env.MONGO_URI;

//------------------------------------------------

// Initialize MongoDB connection
async function initMongoDB() {
    try {
        await mongoose.connect(mongoUri, {
            useNewUrlParser: true,
            useUnifiedTopology: true
        });
        console.log('✅ Connected to MongoDB!');
    } catch (err) {
        console.error("❌ Connection error:", err);
        process.exit(1);
    }
}


// Set up Session Middleware to Track Session Expiration
app.use(session({
    secret: process.env.SESSION_SECRET || 'defaultSecretKey',
    resave: false,
    saveUninitialized: false,
    cookie: {
        maxAge: 1 * 60 * 60 * 1000, // Session expires in 1 hour (1 hour * 60 minutes * 60 seconds * 1000 ms)
        httpOnly: true
    }
}));

// Middleware to make user available in all templates (must be after session middleware)
app.use((req, res, next) => {
    res.locals.user = req.session.user || null;
    next();
});

//------------------------------------------------

// Import routes
const signupRoute = require('./routes/signupRoute');
const loginRoute = require('./routes/loginRoute');
const transcribeRoute = require('./routes/transcribe');

app.use('/signup', signupRoute);
app.use('/login', loginRoute);
app.use('/transcribe', transcribeRoute);

// Route to handle the base page
app.get("/", (req, res) => {
    if (!req.session.isLoggedIn) {
        return res.sendFile(path.join(__dirname, '../Frontend/views', 'index.html'));
    } else {
        return res.sendFile(path.join(__dirname, '../Frontend/views', 'home.html'));
    }
});

// Route to handle the home page
app.get("/home", (req, res) => {
    if (!req.session.isLoggedIn) {
        return res.redirect('/index.html'); // If not logged in, go to index.html
    }
    return res.sendFile(path.join(__dirname, '../Frontend/views', 'home.html')); // Serve home.html if logged in
});

//------------------------------------------------

// Start Express Server AFTER DB Connection
initMongoDB().then(() => {
    // Start the server
    app.listen(port, () => {
        console.log(`🚀 Server running on http://localhost:${port}`);
    });

});
