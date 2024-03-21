import React, { useLayoutEffect } from "react";
import { useState } from "react";
import { useEffect } from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Axios from 'axios';

import "./Navbar.css"


function Navbar(props){

    return (
        <div id="navbar-wrapper" className="flex row align-center justify-between">
            <div id='nav-left'>
                <h1>Legal Search Engine</h1>
            </div>

            <div id="nav-right" className="flex row align-center justify-evenly">
                <h2>About</h2>
                <h2>Documentation</h2>
            </div>
            
        </div>
    )
}

export default Navbar;