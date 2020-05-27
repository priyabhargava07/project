import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
host_address = "6b2f3e65dd6435"
host_pass = "37a854b4f4c4d4"
guest_address = "priyabhargava07@gmail.com"
subject = "Regarding Success of your model "
content = '''Hello, 
				Developer used the commit and it is successfully run . The mail is sent No failed comes up
			THANK YOU ...'''
message = MIMEMultipart()
message['From'] = host_address
message['To'] = guest_address
message['Subject'] = subject
message.attach(MIMEText(content, 'plain'))
session = smtplib.SMTP('smtp.mailtrap.io', 2525)
session.starttls()
session.login(host_address, host_pass)
text = message.as_string()
session.sendmail(host_address, guest_address  , text)
session.quit()
print('Successfully sent your mail')


