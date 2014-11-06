<?php
$this->pageTitle=Yii::app()->name . ' - About';
$this->breadcrumbs=array(
	'Backup',
);
?>
<?php $this->setPageTitle ('Realizar o restaurar copia de seguridad de Base de Datos'); ?>




    <?php
        echo CHtml::link("Descargar Backup Completo", array('/sedes/backup','action'=>1));
    ?>
<br/><hl/>
<form enctype="multipart/form-data" action="/index.php/sedes/backup?action=2" method="POST">
<input type="hidden" name="MAX_FILE_SIZE" value="100000000" />
<b>Seleccione el archivo de backup a Cargar<br/>(TODOS LOS DATOS ACTUALES SE PERDERÁN):</b><br/><input name="uploadedfile" type="file" /><br />
<input type="submit" value="Cargar Backup COMPLETO" />
</form>
<br/>


