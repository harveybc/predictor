<?php $this->pageTitle=Yii::app()->name; ?>


<?php
$this->pageTitle=Yii::app()->name . ' - Administrar';
$this->breadcrumbs=array(
	'Administrar'
); ?>
<h><b>Por favor seleccione uno de los siguientes items para realizar operaciones (crear, borrar, editar, buscar o listar):</b></h>
        
	 <br></br><a href="/index.php/termotablero/admin" >Informes de Tableros Eléctricos</a>
	<br></br><a href="/index.php/termomotores/admin">Motores</a>
    