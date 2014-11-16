/*-----------------------------------------------------------------------------------*/
/*	LOADER
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

$('body').prepend('<div id="spinningSquaresG"><div id="spinningSquaresG_1" class="spinningSquaresG"></div><div id="spinningSquaresG_2" class="spinningSquaresG"></div><div id="spinningSquaresG_3" class="spinningSquaresG"></div><div id="spinningSquaresG_4" class="spinningSquaresG"></div><div id="spinningSquaresG_5" class="spinningSquaresG"></div><div id="spinningSquaresG_6" class="spinningSquaresG"></div><div id="spinningSquaresG_7" class="spinningSquaresG"></div><div id="spinningSquaresG_8" class="spinningSquaresG"></div></div>');
});

jQuery(window).load(function($){

	jQuery('body').find('#spinningSquaresG').remove();
	jQuery('.content').animate({ 'opacity' : '1' }, 500);

});
/*-----------------------------------------------------------------------------------*/
/*	MOBILE NAV
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

	$('#mobile-nav').click(function(){
		$("html, body").animate({ scrollTop: 0 }, 200);
		setTimeout(function(){
			$('header').toggleClass('active');
			$('#mobile-nav').toggleClass('active');	
		}, 200);	
	});
	
});
/*-----------------------------------------------------------------------------------*/
/*	NAVIGATION ACTIVE
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

	$('nav a[href^="' + location.pathname.split("/")[1] + '"]').addClass('active').parents('li').children('a').addClass('active');
	
});
/*-----------------------------------------------------------------------------------*/
/*	SLIDER
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

	$(".rslides").responsiveSlides({
	  speed: 500,
	  timeout: 4000,
	  pager: true
	});
	
});
/*-----------------------------------------------------------------------------------*/
/*	ISOTOPE
/*-----------------------------------------------------------------------------------*/
jQuery(window).load(function($){
'use strict';

	jQuery('ul.grid').isotope({
		itemSelector : 'li',
		transformsEnabled : false
	});
	
	jQuery('.filters a').click(function(){
		var filter = jQuery(this).attr('data-href');
		jQuery('.filters li').removeClass('active');
		jQuery(this).parent().addClass('active');
		jQuery('ul.grid').isotope({ filter: filter });
		jQuery(window).trigger('resize');
		return false;
	});
	
	jQuery(window).smartresize(function(){
		jQuery('ul.grid').isotope('reLayout');
	});
	
	jQuery(window).trigger('resize');
	
	jQuery('header').height( jQuery(document).height() );
	
	jQuery(window).resize(function(){
		jQuery('header').height( jQuery(window).height() );
		setTimeout(function(){
			jQuery('header').height( jQuery(document).height() );
		}, 900);
	});
	
});
/*-----------------------------------------------------------------------------------*/
/*	HOVER DIR
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

	$(function(){
		$('ul.grid.portfolio li, .more-hover').each( function() { $(this).hoverdir(); } );
	});

});
/*-----------------------------------------------------------------------------------*/
/*	GALLERY HOVER
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

	$('.gallery.animate li').hover(function(){
		$('.gallery li').not(this).stop().animate({ 'opacity' : '0.3' }, 200);
	}, function(){
		$('.gallery li').stop().animate({ 'opacity' : '1' }, 200);
	});
});
/*-----------------------------------------------------------------------------------*/
/*	AJAX PORTFOLIO
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';
	
	$('body').on('click', 'ul.grid li a, .gallery-wrapper a', function(){
		var url = $(this).attr('href');
		
		$('.content').prepend('<div id="spinningSquaresG"><div id="spinningSquaresG_1" class="spinningSquaresG"></div><div id="spinningSquaresG_2" class="spinningSquaresG"></div><div id="spinningSquaresG_3" class="spinningSquaresG"></div><div id="spinningSquaresG_4" class="spinningSquaresG"></div><div id="spinningSquaresG_5" class="spinningSquaresG"></div><div id="spinningSquaresG_6" class="spinningSquaresG"></div><div id="spinningSquaresG_7" class="spinningSquaresG"></div><div id="spinningSquaresG_8" class="spinningSquaresG"></div></div>');
		
		$.get(url, function(data){
			var filtered = jQuery(data).filter('section').removeClass('content');
			
			$(filtered).find(".rslides").responsiveSlides({
			  speed: 500,
			  timeout: 4000,
			  pager: true
			});
			
			$(filtered).imagesLoaded(function(){
				$('ul.grid, .gallery-wrapper').animate({ 'left' : '-100%', 'opacity' : '0' }, function(){
					$('ul.grid, .gallery-wrapper').css('max-height', '0px');
					$("html, body").animate({ scrollTop: 0 }, 200);
					$('#loader').html(filtered);
					$('#spinningSquaresG').remove();
					$('#loader').slideDown(function(){
						jQuery(window).trigger('resize');
					});
				});
			});
		});
		return false;
	});
	
	$('body').on('click', '.post-nav', function(){
		var url = $(this).attr('href');
		
		$('.content').prepend('<div id="spinningSquaresG"><div id="spinningSquaresG_1" class="spinningSquaresG"></div><div id="spinningSquaresG_2" class="spinningSquaresG"></div><div id="spinningSquaresG_3" class="spinningSquaresG"></div><div id="spinningSquaresG_4" class="spinningSquaresG"></div><div id="spinningSquaresG_5" class="spinningSquaresG"></div><div id="spinningSquaresG_6" class="spinningSquaresG"></div><div id="spinningSquaresG_7" class="spinningSquaresG"></div><div id="spinningSquaresG_8" class="spinningSquaresG"></div></div>');
		
		$.get(url, function(data){
			var filtered = jQuery(data).filter('section').removeClass('content');
			
			$(filtered).find(".rslides").responsiveSlides({
			  speed: 500,
			  timeout: 4000,
			  pager: true
			});
			
			$(filtered).imagesLoaded(function(){
				$('#loader').animate({ 'left' : '-100%', 'opacity' : '0' }, function(){
					$("html, body").animate({ scrollTop: 0 }, 200);
					$('#spinningSquaresG').remove();
					$('#loader').html(filtered).animate({ 'left' : '0', 'opacity' : '1' });
				});
			});
		});
		return false;
	});
	
	$('body').on('click', 'a.close', function(){
		
		$('#loader').slideUp(function(){
			$('ul.grid, .gallery-wrapper').css('max-height', '');
			$('ul.grid, .gallery-wrapper').animate({ 'left' : '0', 'opacity' : '1' },function(){
				jQuery(window).trigger('resize');
			});
		});
		
		return false;
	});

});
/*-----------------------------------------------------------------------------------*/
/*	VERTICAL GALLERY
/*-----------------------------------------------------------------------------------*/
jQuery(window).load(function($){
'use strict';

if(jQuery('#vertical').length > 0){
	var sly = new Sly(jQuery('#vertical'), {
		horizontal: 1,
		itemNav: 'basic',
		smart: 1,
		activateOn: 'click',
		mouseDragging: 1,
		touchDragging: 1,
		releaseSwing: 1,
		startAt: 0,
		scrollBy: 1,
		activatePageOn: 'click',
		speed: 300,
		elasticBounds: 1,
		dragHandle: 1,
		dynamicHandle: 1,
		clickBar: 1,
	}).init();
	
	jQuery(window).resize(function(){
		sly.reload();
	});
}
	
});
/*-----------------------------------------------------------------------------------*/
/*	VEIW BACKGROUND
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

$('.view-background').click(function(){
	if( $('.content').css('left') == '0px' ){
		$('.content').animate({ 'left' : '-100%', 'opacity' : '0' });
		$('.view-background').html('<i class="icon-eye-open icon-2x"></i>');
	} else {
		$('.content').animate({ 'left' : '0', 'opacity' : '1' });
		$('.view-background').html('<i class="icon-eye-close icon-2x"></i>');
	}
	return false;
});
	
});
/*-----------------------------------------------------------------------------------*/
/*	TABS
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

		$('.tab-container').easytabs();

});
/*-----------------------------------------------------------------------------------*/
/*	ALERTS
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

		$('.alert i').click(function(){
			$(this).parent().slideUp();
		});

});
/*-----------------------------------------------------------------------------------*/
/*	ACCORDION
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

		$('.accordion > dd.active').show();
		  
		$('.accordion > dt > a').click(function() {
			if( $(this).parent().hasClass('active') ){
				$(this).parents('.accordion').find('dt').removeClass('active');
				$(this).parents('.accordion').find('dd').removeClass('active').slideUp();
				return false;
			} else {
				$(this).parents('.accordion').find('dt').removeClass('active');
				$(this).parents('.accordion').find('dd').removeClass('active').slideUp();
				$(this).parent().addClass('active').next().addClass('active').slideDown();
				return false;
			}
		});

});
/*-----------------------------------------------------------------------------------*/
/*	CONTACT FORM
/*-----------------------------------------------------------------------------------*/
jQuery(document).ready(function($){
'use strict';

	//CONTACT FORM
		$('#contactform').submit(function(){
	
			var action = $(this).attr('action');
	
			$("#message").slideUp(750,function() {
			$('#message').hide();
	
	 		$('#submit').attr('disabled','disabled');
	
			$.post(action, {
				name: $('#name').val(),
				email: $('#email').val(),
				website: $('#website').val(),
				comments: $('#comments').val()
			},
				function(data){
					document.getElementById('message').innerHTML = data;
					$('#message').slideDown('slow');
					$('#submit').removeAttr('disabled');
					if(data.match('success') != null) $('#contactform').slideUp('slow');
					$(window).trigger('resize');
				}
			);
	
			});
	
			return false;
	
		});
	
});