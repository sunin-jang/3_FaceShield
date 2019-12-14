$(document).ready( function() {
    let slide = $('#mainmenu li:nth-child(2)');
    slide.on('mouseover', function() {
        $(this).children("li").css({"display":"inline-block"});
    });
    slide.on('mouseleave', function() {
        $(this).children("li").css({"display":"none"});
    });
});