<?php
$this->pageTitle=Yii::app()->name . ' - Buscar Orden de Trabajo';
$this->breadcrumbs=array(
	'Buscar OT',
);
?>
<?php $this->setPageTitle ('Buscar Orden de Trabajo'); ?>
<div class="form">
<b>Por favor, ingrese la orden de trabajo y oprima Aceptar</b><br/>
<?php echo CHtml::form("", 'get', array ( )); ?>

<?php
// Si existe el parámetro $_GET[OT]
echo CHtml::textField('OT', isset($_GET['OT'])? $_GET['OT']:"",array ( )); ?>
<div class="row buttons">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>
<?php 
//crea el formulario
echo CHtml::endForm(); ?>
    
<?php 
    // si existe el parámetro OT
    if (isset($_GET['OT'])){
        echo '<fieldset style="padding-left:15px;padding-bottom:15px;width:600px">';
        echo '<legend>Resultados de la búsqueda</legend>';
        $encontrado=0;
        $ruta=array();
        // busca en cada modelo la OT
        // Aceitesnivel1 
        $model=Aceitesnivel1::model()->findAllByAttributes(array('OT'=>$_GET['OT']));
        foreach($model as $foreignobj) {
            array_push($ruta, '/index.php/aceitesnivel1/view/'.$foreignobj->id);
        }
        // Aislamiento_acometida 
        $model=Aislamiento_acometida::model()->findAllByAttributes(array('OT'=>$_GET['OT']));
        foreach($model as $foreignobj) {
            array_push($ruta, '/index.php/aislamiento_acometida/view/'.$foreignobj->Toma);
        }
        // Aislamiento_fases 
        $model=Aislamiento_fases::model()->findAllByAttributes(array('OT'=>$_GET['OT']));
        foreach($model as $foreignobj) {
            array_push($ruta, '/index.php/aislamiento_fases/view/'.$foreignobj->Toma);
        }
        // Aislamiento_tierra
        $model=Aislamiento_tierra::model()->findAllByAttributes(array('OT'=>$_GET['OT']));
        foreach($model as $foreignobj) {
            array_push($ruta, '/index.php/aislamiento_tierra/view/'.$foreignobj->Toma);
        }
        // Reportes
        $model=Reportes::model()->findAllByAttributes(array('OT'=>$_GET['OT']));
        foreach($model as $foreignobj) {
            array_push($ruta, '/index.php/reportes/view/'.$foreignobj->id);
        }
        // Termomotores
        $model=Termomotores::model()->findAllByAttributes(array('OT'=>$_GET['OT']));
        foreach($model as $foreignobj) {
            array_push($ruta, '/index.php/termomotores/view/'.$foreignobj->id);
        }
        // Termotablero
        $model=Termotablero::model()->findAllByAttributes(array('OT'=>$_GET['OT']));
        foreach($model as $foreignobj) {
            array_push($ruta, '/index.php/termotablero/view/'.$foreignobj->id);
        }
        // Vibraciones
        $model=Vibraciones::model()->findAllByAttributes(array('OT'=>$_GET['OT']));
        foreach($model as $foreignobj) {
            array_push($ruta, '/index.php/vibraciones/view/'.$foreignobj->id);
        }
        // si la encontró, renderiza la Detalles de donde se encontró.
        foreach($ruta as $foreignobj) {
            echo "<br/>".CHtml::link($foreignobj,$foreignobj);
        }
        echo "</fieldset>";
    }
        
?>



</div>
