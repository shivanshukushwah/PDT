# PDT
a tool that detects phishing attempts by analyzing URLs and email content.
The Phishing Detection Tool is a simple, easy-to-use program designed to help identify potentially harmful phishing websites. It utilizes a machine learning model to analyze key characteristics of a URL and predict whether it is legitimate or malicious. Here's how it works:

Extract Features from URLs: The tool examines various properties of a URL, such as:

The length of the URL.
The number of dots (.) in the URL, indicating subdomains.
The presence of special characters like hyphens (-) or slashes (/).
Whether the URL uses a secure connection (HTTPS).
The number of subdomains, which can be a sign of suspicious activity.
Train a Machine Learning Model: Using example data, the tool trains a Random Forest Classifier, a robust and widely used algorithm, to distinguish between phishing and legitimate URLs.

Evaluate and Save the Model: The program evaluates the model's performance on test data and saves the trained model for future use.

Real-Time Predictions: You can input any URL, and the tool will analyze its features and predict whether it's likely to be phishing or legitimate.

Key Benefits:
User-Friendly: The tool is straightforward to run and provides clear results.
Adaptable: It can be improved with larger, real-world datasets for better accuracy.
Educational: Understand how phishing URLs differ from legitimate ones through feature analysis.
This tool is a helpful resource for cybersecurity enthusiasts, educators, or anyone wanting to enhance their awareness and protection against online threats.
