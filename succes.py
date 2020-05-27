import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
host_address = "9d8e7540a04f8a"
host_pass = "d788d7303e7936"
guest_address = "shubhamkhandelwal523@gmail.com"
subject = "Regarding Success of your model "
content = '''Hello, 
				Developer used the commit and it is successfully run . The mail is sent No failed comes up
			THANK YOU ...'''
message = MIMEMultipart()
message['From'] = host_address
message['To'] = guest_address
message['Subject'] = subject
message.attach(MIMEText(content, 'plain'))
session = smtplib.SMTP('smtp.mailtrap.io', 587)
session.starttls()
session.login(host_address, host_pass)
text = message.as_string()
session.sendmail(host_address, guest_address  , text)
session.quit()
print('Successfully sent your mail')


