# Google login setup for Accredix

The `.exe` includes the Google OAuth code, but Google will reject sign-in until
the app has a Google OAuth web application client.

1. In Google Cloud Console, create or select a project.
2. Configure the OAuth consent screen for the intended users.
3. Create an OAuth client ID with application type `Web application`.
4. Add this exact authorized redirect URI:

   `http://127.0.0.1:5000/auth/callback`

5. Set these environment variables before launching Accredix:

```powershell
$env:ACCREDITX_SECRET_KEY = "use-a-long-random-secret"
$env:GOOGLE_CLIENT_ID = "your-client-id.apps.googleusercontent.com"
$env:GOOGLE_CLIENT_SECRET = "your-client-secret"
& ".\Accredix.exe"
```

Without these values, use **Enter local demo workspace**. That button is the
working offline path and does not pretend to be Google authentication.
