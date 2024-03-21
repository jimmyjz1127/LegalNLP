import logo from './logo.svg';
import './App.css';

import HomePage from './Components/HomePage/HomePage';

import './base.css'

import Navbar from './Components/Navbar/Navbar'

function App() {
  return (
    <div className="App">
      <Navbar/>
      <HomePage/>
    </div>
  );
}

export default App;
