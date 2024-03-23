import React from "react";
import { useState } from "react";
import { useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Axios from 'axios';

import Navbar from './../Navbar/Navbar'
import Result from './../Result/Result'

import LoadingIcon from './../../Assets/loading.gif'

import "./HomePage.css"

function HomePage(props){

    const [results, setResults] = useState([]);
    const [numResults, setNumResults] = useState(0);
    const [query, setQuery] = useState("");
    const [loading, setLoading] = useState(0)

    const [searchType, setSearchType] = useState()
    const [selectedOption, setSelectedOption] = useState('');

    const searchQuery = async () => {
        setLoading(1)
        setNumResults(0)
        try {
            const res = await Axios({
                method:'POST',
                withCredentials:true,
                data:{
                    query:query
                },
                url:'http://localhost:5000/search'
            })

            const data = JSON.parse(res.data.data)
            console.log(data)
            setResults(data)
            setNumResults(data.length)
            setLoading(0)
        } catch (err) {
            console.log(err)
            setLoading(0)
        }
    }

    useEffect(() => {
        const fetchData = async () => {
            try {
                let res = await Axios({
                    method: 'GET',
                    withCredentials: true,
                    data: {}, // Note: GET requests generally do not have a body. You might want to remove this if it's not needed.
                    url: 'http://localhost:5000/searchType'
                });
                let type = res.data.searchType;
                setSearchType(type);
                if (type === '1') {
                    setSelectedOption('Cross Encoder');
                } else {
                    setSelectedOption('Dual Encoder');
                }
            } catch (err) {
                console.log(err);
            }
        };
    
        fetchData();
    }, []);
    

    const changeEncoding = async () => {
        let type = '2'

        if (document.getElementById('encoding-dropdown').value == 'Cross Encoder')  {
            type = '1'
        }
        if (type != searchType ){
            try {
                let res = await Axios({
                    method:'POST',
                    withCredentials:true,
                    data:{
                        'searchType' : type
                    },
                    url:'http://localhost:5000/changeSearchType'
                })
                console.log(type)
                setSearchType(type)
                // if (type == '1')  setSelectedOption('Cross Encoder')
                // else setSelectedOption('Dual Encoder')
                // console.log(selectedOption)
            } catch (err){
                console.error(err)
            }
        }
    }

    return (
        <div id="home-wrapper" className="flex col align-center justify-center">
            <div id='search-bar-modal' className="flex row align-center">
                <input type='text' placeholder="Enter query..." id="search-bar-input" onChange={(e) => setQuery(e.target.value)}/>
                <button id="search-btn" onClick={(e) => searchQuery()}>Search</button>

                <select id="encoding-dropdown" onChange={(e) => changeEncoding()}>
                    <option >Cross Encoder</option>
                    <option >Dual Encoder</option>
                </select>

            </div>
            <div id='results-wrapper'>
                <div id='results-top-bar' className="flex row align-center justify-between">
                    <h2>Result Summary</h2>
                    <div className="flex row align-center">Number of Results : <div id='result-num'>{numResults}</div></div>
                </div>

                <div id="results-main" className="flex col align-start">
                    {loading ?
                        <div id='loader-wrapper' class="flex col align-center justify-center">
                            <img src={LoadingIcon} id='result-loader'/>
                            <h2>Loading...</h2>
                        </div>:
                        
                        results.map((result, index) => {
                            return (
                                <Result data={result} index={index} key={index}/>
                            );
                        })
                        
                    }
                    
                </div>

                <div id='results-bottom-bar' className="flex row align-center">
                    <button className="default-btn" id="download-btn">Download</button>
                </div>
            </div>
        </div>
    ) ;
}

export default HomePage;