<?php

if(!$_POST) exit;

function tommus_email_validate($email) { return filter_var($email, FILTER_VALIDATE_EMAIL) && preg_match('/@.+\./', $email); }

$name = $_POST['name']; $email = $_POST['email']; $website = $_POST['website']; $comments = $_POST['comments'];


if(trim($name) == '') {
	exit('<div class="alert danger">Attention! You must enter your name.</div>');
} else if(trim($name) == 'Name') {
	exit('<div class="alert danger">Attention! You must enter your name.</div>');
} else if(trim($email) == '') {
	exit('<div class="alert danger">Attention! Please enter a valid email address.</div>');
} else if(!tommus_email_validate($email)) {
	exit('<div class="alert danger">Attention! You have entered an invalid e-mail address.</div>');
} else if(trim($website) == 'Website') {
	exit('<div class="alert danger">Attention! Please enter your website.</div>');
} else if(trim($website) == '') {
	exit('<div class="alert danger">Attention! Please enter your website.</div>');
} else if(trim($comments) == 'Message') {
	exit('<div class="alert danger">Attention! Please enter your message.</div>');
} else if(trim($comments) == '') {
	exit('<div class="alert danger">Attention! Please enter your message.</div>');
} else if( strpos($comments, 'href') !== false ) {
	exit('<div class="alert danger">Attention! Please leave links as plain text.</div>');
} else if( strpos($comments, '[url') !== false ) {
	exit('<div class="alert danger">Attention! Please leave links as plain text.</div>');
} if(get_magic_quotes_gpc()) { $comments = stripslashes($comments); }

//ENTER YOUR EMAIL ADDRESS HERE
$address = 'hello@email.com';

$e_subject = 'You\'ve been contacted by ' . $name . '.';
$e_body = "You have been contacted by $name from $website from your contact form, their additional message is as follows." . "\r\n" . "\r\n";
$e_content = "\"$comments\"" . "\r\n" . "\r\n";
$e_reply = "You can contact $name $last via email, $email";

$msg = wordwrap( $e_body . $e_content . $e_reply, 70 );

$headers = "From: $address" . "\r\n";
$headers .= "Reply-To: $email" . "\r\n";
$headers .= "MIME-Version: 1.0" . "\r\n";
$headers .= "Content-type: text/plain; charset=utf-8" . "\r\n";
$headers .= "Content-Transfer-Encoding: quoted-printable" . "\r\n";



if(mail($address, $e_subject, $msg, $headers)) { echo "<fieldset><div id='success_page'><h3 class='remove-bottom'>Email Sent Successfully.</h3><p>Thank you $name, your message has been submitted to us.</p></div></fieldset>"; }