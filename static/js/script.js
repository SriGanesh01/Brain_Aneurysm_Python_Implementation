// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Toggle table output visibility on table title click
    const tableTitles = document.querySelectorAll('.table-title');
    tableTitles.forEach(function(title) {
        title.addEventListener('click', function() {
            const tableOutput = this.nextElementSibling;
            // Toggle using inline style as well as an "active" class for transitions
            if (tableOutput.style.display === "none" || tableOutput.style.display === "") {
                tableOutput.style.display = "block";
            } else {
                tableOutput.style.display = "none";
            }
            // Toggle active class for smooth CSS transitions
            tableOutput.classList.toggle('active');
        });
    });

    // Add tooltip on plot title hover
    const plotTitles = document.querySelectorAll('.plot-title');
    plotTitles.forEach(function(title) {
        title.addEventListener('mouseover', function(event) {
            // Check if a tooltip already exists
            if (this.querySelector('.tooltip')) return;
            
            const tooltip = document.createElement('span');
            tooltip.className = 'tooltip';
            tooltip.textContent = "Hover on the image for interactive effects";
            this.appendChild(tooltip);
        });
        
        title.addEventListener('mouseout', function(event) {
            const tooltip = this.querySelector('.tooltip');
            if (tooltip) {
                tooltip.remove();
            }
        });
    });

    // Additional interactivity for plot images
    const plotImages = document.querySelectorAll('.hover-image');
    plotImages.forEach(function(image) {
        // Add tooltip when hovering over the image
        image.addEventListener('mouseover', function(){
            // Check if the tooltip already exists in the parent container
            if (!this.parentElement.querySelector('.image-tooltip')) {
                const imgTooltip = document.createElement('span');
                imgTooltip.className = 'image-tooltip';
                imgTooltip.textContent = "Double click to expand";
                this.parentElement.appendChild(imgTooltip);
            }
        });
        image.addEventListener('mouseout', function(){
            const imgTooltip = this.parentElement.querySelector('.image-tooltip');
            if (imgTooltip) {
                imgTooltip.remove();
            }
        });

        // Expand image in a modal on double click
        image.addEventListener('dblclick', function() {
            // Create modal overlay
            const modalOverlay = document.createElement('div');
            modalOverlay.className = 'modal-overlay';
            const modalContent = document.createElement('div');
            modalContent.className = 'modal-content';
            const modalImage = document.createElement('img');
            modalImage.src = this.src;
            modalImage.className = 'modal-image';
            modalContent.appendChild(modalImage);
            modalOverlay.appendChild(modalContent);
            document.body.appendChild(modalOverlay);
            
            // Close modal when clicking on overlay
            modalOverlay.addEventListener('click', function(){
                document.body.removeChild(modalOverlay);
            });
        });
    });
});
