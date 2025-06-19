import React from 'react';

window.addEventListener('load', () => {
  localStorage.clear();
});

function Tagline() {
  return (
    <div className="tagline">
      Cogno AI. Your insurance guide, always by your side.
    </div>
  );
}

export default Tagline;
