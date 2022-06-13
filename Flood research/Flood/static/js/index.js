const citymap = {
  chicago: {
    center: { lat: 6.9061, lng: 79.9696 },
    population: 100,
  }
};

function initMap() {
  // Create the map.
  const map = new google.maps.Map(document.getElementById("map"), {
    zoom: 15,
    center: { lat: 6.9061, lng: 79.9696 },
    mapTypeId: "terrain",
  });

  // Construct the circle for each value in citymap.
  // Note: We scale the area of the circle based on the population.
  for (const city in citymap) {
    // Add the circle for this city to the map.
    const cityCircle = new google.maps.Circle({
      strokeColor: "#FF0000",
      strokeOpacity: 0.8,
      strokeWeight: 2,
      fillColor: "#FF0000",
      fillOpacity: 0.35,
      map,
      center: citymap[city].center,
      radius: Math.sqrt(citymap[city].population) * 100,
    });
  }
}