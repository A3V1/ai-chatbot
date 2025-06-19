import { render, screen } from '@testing-library/react';
import App from './App';

window.addEventListener('load', () => {
  localStorage.clear();
});

test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
