<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/MLResearchAtOSRAM/OsCINN">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h1 align="center">OScINN</h1>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/MLResearchAtOSRAM/OsCINN"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/MLResearchAtOSRAM/OsCINN">View Demo</a>
    ·
    <a href="https://github.com/MLResearchAtOSRAM/OsCINN/issues">Report Bug</a>
    ·
    <a href="https://github.com/MLResearchAtOSRAM/OsCINN/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

The OScINN implements the invertible- and conditional neural networks from the paper "Investigation of inverse design of multilayer thin-films with conditional invertible Neural Networks"[[1](#1),[2](#2)]. The architecture is heavily based on the contribution of Ardizzone et al [3](#3) who published the [FrEIA]() package which enabled our work.

The Repo contains only the architecture for the network, which was evaluated in the aforementioned paper but also contains a short *Introduction* jupyter notebook which gives an example how to use the networks.

For replicating the results of the paper, the thin-film computation/dataset generation can be done via the TMM-Fast(https://github.com/MLResearchAtOSRAM/tmm_fast) package, which also contains convenience routines for dataset generation. Have fun!

Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `MLResearchAtOSRAM`, `OsCINN`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

You only need a running Python environment with Python >= 3.7. Its recommended to use Conda to set up an environment 

### Installation

1. Clone the repo by typing into your prompt
   ```sh
   git clone https://github.com/MLResearchAtOSRAM/OsCINN.git
   ```
2. Install requirements via pip 
   ```sh
   pip install -r requirements.txt
   ```
3. Train some Network!


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
Thanks to
* [Heribert Wankerl](https://github.com/HarryTheBird) for contributing to the development 
* [Maike Stern](https://github.com/MLResearchAtOSRAM) for running the Repo
* [Daniel Grünbaum](https://github.com/dg46) just for fun

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- References -->
## References
<a id="1">[1]</a> 
Luce, A. et. al. (2022). 
Investigation of inverse design of multilayer thin-films with conditional invertible Neural Networks. 
Preprint.

<a id="2">[2]</a> 
Luce, A. et. al. (2022). 
Investigation of inverse design of multilayer thin-films with conditional invertible Neural Networks. 
Publication.

<a id="3">[3]</a> 
Ardizzone, L. et. al. (2020). 
Conditional Invertible Neural Networks for Diverse Image-to-Image Translation. 
https://doi.org/10.48550/arXiv.2105.02104

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[forks-shield]: https://img.shields.io/github/forks/MLResearchAtOSRAM/OsCINN.svg?style=for-the-badge
[forks-url]: https://github.com/MLResearchAtOSRAM/OsCINN/network/members

[stars-shield]: https://img.shields.io/github/stars/MLResearchAtOSRAM/OsCINN.svg?style=for-the-badge
[stars-url]: https://github.com/MLResearchAtOSRAM/OsCINN/stargazers

[issues-shield]: https://img.shields.io/github/issues/MLResearchAtOSRAM/OsCINN.svg?style=for-the-badge
[issues-url]: https://github.com/MLResearchAtOSRAM/OsCINN/issues

[license-shield]: https://img.shields.io/github/license/MLResearchAtOSRAM/OsCINN.svg?style=for-the-badge
[license-url]: https://github.com/MLResearchAtOSRAM/OsCINN/master/LICENSE.txt

[product-screenshot]: images/screenshot.png