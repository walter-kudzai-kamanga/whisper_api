# Email Setup Guide - SendGrid Configuration

This application now uses **SendGrid** as the exclusive email provider for sending transactional emails like welcome messages and password reset requests.

## Setup Instructions

### 1. Create a SendGrid Account
1. Go to [SendGrid.com](https://sendgrid.com)
2. Sign up for a free account (100 emails/day free tier)
3. Verify your account

### 2. Get Your SendGrid API Key
1. Log into your SendGrid dashboard
2. Go to **Settings** → **API Keys**
3. Click **Create API Key**
4. Choose **Full Access** or **Restricted Access** with **Mail Send** permissions
5. Copy the API key (you won't see it again!)

### 3. Verify Your Sender Email
1. Go to **Settings** → **Sender Authentication**
2. Choose either:
   - **Domain Authentication** (recommended for production)
   - **Single Sender Verification** (for testing)
3. Follow the verification steps

### 4. Configure Environment Variables
Add these variables to your `.env` file:

```bash
# SendGrid Email Configuration
SENDGRID_API_KEY=your_sendgrid_api_key_here
FROM_EMAIL=your-verified-sender@yourdomain.com
FROM_NAME=Audio Transcription API
```

### 5. Test Your Configuration
The application will automatically use SendGrid for:
- Welcome emails when users register
- Password reset emails

## Email Templates

The application uses Jinja2 templates located in the `templates/` directory:
- `welcome.html` - Welcome email for new users
- `password_reset.html` - Password reset email

## Troubleshooting

### Common Issues:

1. **"SendGrid API key not configured"**
   - Make sure `SENDGRID_API_KEY` is set in your environment
   - Check that the API key is correct

2. **"Email not sent"**
   - Verify your sender email is authenticated in SendGrid
   - Check SendGrid dashboard for delivery status
   - Ensure your API key has "Mail Send" permissions

3. **Emails going to spam**
   - Use Domain Authentication instead of Single Sender Verification
   - Warm up your IP address gradually
   - Follow email best practices

### SendGrid Dashboard Features:
- **Activity** - View email delivery status
- **Statistics** - Monitor email performance
- **Settings** - Configure webhooks, IP addresses, etc.

## Migration from Other Providers

The following email providers have been removed:
- ❌ Gmail SMTP
- ❌ Mailgun (SMTP and API)
- ❌ Resend
- ❌ FastAPI-Mail

All email functionality now uses SendGrid exclusively for better reliability and deliverability.

## Support

For SendGrid-specific issues:
- [SendGrid Documentation](https://docs.sendgrid.com/)
- [SendGrid Support](https://support.sendgrid.com/)

For application-specific issues, check the application logs for detailed error messages. 