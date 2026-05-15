async function loadViewer() {
  const target = document.getElementById("viewer");
  const response = await fetch("assets/reconstruction_points.json");
  const points = await response.json();

  const trace = {
    type: "scatter3d",
    mode: "markers",
    x: points.x,
    y: points.y,
    z: points.z,
    marker: {
      size: 3,
      color: points.occupancy,
      colorscale: [
        [0, "#1d5f73"],
        [0.5, "#e1b55d"],
        [1, "#b84e35"]
      ],
      opacity: 0.82,
      colorbar: {
        title: "occupancy",
        thickness: 12
      }
    },
    hovertemplate: "x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<br>occ=%{marker.color:.3f}<extra></extra>"
  };

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    scene: {
      aspectmode: "data",
      xaxis: { title: "x", showbackground: false, gridcolor: "#d8d1c3" },
      yaxis: { title: "y", showbackground: false, gridcolor: "#d8d1c3" },
      zaxis: { title: "z", showbackground: false, gridcolor: "#d8d1c3" },
      camera: { eye: { x: 1.45, y: 1.25, z: 0.9 } }
    }
  };

  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  };

  Plotly.newPlot(target, [trace], layout, config);
}

loadViewer().catch((error) => {
  const target = document.getElementById("viewer");
  target.textContent = `Viewer failed to load: ${error.message}`;
});
