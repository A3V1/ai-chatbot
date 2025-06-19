import React from 'react';

window.addEventListener('load', () => {
  localStorage.clear();
});

function Header() {
  return (
    <header>
      <nav>
        <div className="logo">Cogno AI</div>
        <ul>
          <li><a href="#">Home</a></li>
          <li><a href="#">Plans</a></li>
          <li><a href="#">Claims</a></li>
          <li><a href="#">Support</a></li>
          <li><a href="#">About</a></li>
        </ul>
      </nav>
    </header>
  );
}

export default Header;
