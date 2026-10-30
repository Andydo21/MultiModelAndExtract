document.querySelectorAll('.contract-card').forEach((card, index) => {
    card.style.animation = `cardIn .35s ease ${index * 35}ms both`;
});
const style = document.createElement('style');
style.textContent = '@keyframes cardIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}';
document.head.appendChild(style);
