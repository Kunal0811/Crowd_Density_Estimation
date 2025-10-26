// Smooth scroll for nav links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e){
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if(target) target.scrollIntoView({behavior:'smooth', block:'start'});
    });
});

// Navbar background change on scroll
window.addEventListener('scroll', function(){
    const nav = document.querySelector('nav');
    nav.style.background = window.scrollY > 50 ? 'rgba(0,0,0,0.9)' : 'rgba(255,255,255,0.1)';
});

// Intersection Observer for fade-in animations
const observerOptions = { threshold: 0.1, rootMargin: '0px 0px -50px 0px' };
const observer = new IntersectionObserver((entries)=>{
    entries.forEach(entry=>{
        if(entry.isIntersecting){
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
});

document.querySelectorAll('.card-hover, section').forEach(el=>{
    el.style.opacity='0';
    el.style.transform='translateY(30px)';
    el.style.transition='all 0.6s ease-out';
    observer.observe(el);
});

// Button hover animation
document.querySelectorAll('button').forEach(button=>{
    button.addEventListener('mouseenter', ()=> button.style.transform='translateY(-2px)');
    button.addEventListener('mouseleave', ()=> button.style.transform='translateY(0)');
});

// Dynamic live stats update
function updateStats(){
    const peopleCount = document.querySelector('.floating-animation .text-yellow-400');
    const accuracy = document.querySelector('.floating-animation .text-green-400');
    const density = document.querySelector('.floating-animation .text-blue-400');

    if(peopleCount){
        let currentCount = parseInt(peopleCount.textContent);
        peopleCount.textContent = Math.max(0, currentCount + Math.floor(Math.random()*10-5));
    }
    if(accuracy){
        let currentAcc = parseInt(accuracy.textContent);
        accuracy.textContent = Math.min(100, Math.max(70, currentAcc + Math.floor(Math.random()*6-3))) + '%';
    }
    if(density){
        let currentDens = parseFloat(density.textContent);
        density.textContent = Math.max(0, currentDens + (Math.random()*2-1)).toFixed(1);
    }
}
setInterval(updateStats, 3000);
