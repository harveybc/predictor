<?php $this->pageTitle=Yii::app()->name; ?>


<?php
$this->pageTitle=Yii::app()->name . ' - Administrar';
$this->breadcrumbs=array(
	'Administrar',
); ?>
<h><b>Por favor seleccione uno de los siguientes items para realizar operaciones (crear, borrar, editar, buscar o listar):</b></h>
        
<br></br><a href="/index.php/site/page?view=instrucciones">Instrucciones Operación Equipos Mantenimiento</a>
        <br></br><a href="/index.php/verificacionEquipos/index" >Verificación Equipos CBM</a>
	