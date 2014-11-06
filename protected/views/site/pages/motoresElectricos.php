
<?php $this->pageTitle=Yii::app()->name; ?>


<?php
$this->pageTitle=Yii::app()->name . ' - Administrar';
$this->breadcrumbs=array(
	'Administrar',
); ?>
<h><b>Por favor seleccione uno de los siguientes items para realizar operaciones:</b></h>

       
	<br></br><a href="/index.php/aislamiento_tierra/admin">Registro Aislamiento</a>
	<br></br><a href="/index.php/vibraciones/admin">Registro Vibraciones y Temperatura</a>
        <br></br><a href="/index.php/termomotores/admin">Termografía de motores</a>
        <br></br><a href="/index.php/site/page?view=resumen" >Resumen de Resultados</a>
	<br></br><a href="/index.php/Aceitesnivel1/admin">Lubricantes</a>
	
