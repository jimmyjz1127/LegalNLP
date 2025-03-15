import React, { useEffect } from 'react';
import Plot from 'react-plotly.js'

const AttentionVisual = (props) => {
	const {query, attentionData} = props;


	// Assuming `attentionData` is a 2D array passed in as props to this component
	return (
		<div>
			<Plot
				data={[
					{
						z: attentionData,
						type: 'heatmap',
						colorscale: 'Viridis',
					}
				]}
				layout={{
					title: 'Query Attention Weights Heatmap',
					xaxis: { 
						title: 'Token Index',
						tickmode:'array',
						tickvals: query.map((_, index) => index),
						ticktext: query
					},
					yaxis: { 
						title: 'Token Index',
						tickmode:'array',
						tickvals: query.map((_, index) => index),
						ticktext: query
					},
					width: 400, // Customize the size as needed
					height: 400,
				}}
			/>
		</div>
		
	);
};

export default AttentionVisual;