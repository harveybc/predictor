<?php
$model = Usuarios::model()->findbypk(0 + $id);
if (0 + $panel == 1) {
    
    
           echo '<a href="/index.php/usuarios/create">Adicionar Usuario </a> <br/>';
    
    echo"<ul>";
    foreach($model->responsabilidades as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->idClinica0->nombre, array('responsabilidades/view', 'id' => $foreignobj->id)));

				}
    echo"</ul>";
}
if (0 + $panel == 2) {
    
    
           echo '<a href="/index.php/visitas/create">Adicionar Visita </a> <br/>';
    
    echo "<ul>";
    foreach($model->visitases as $foreignobj) { 

				printf('<li>%s - %s</li>', CHtml::link($foreignobj->fecha, array('visitas/view', 'id' => $foreignobj->id)),
                                        CHtml::link($foreignobj->idClinica0->nombre, array('visitas/view', 'id' => $foreignobj->id))
                                );

				}
    echo "</ul>";
}
if (0 + $panel == 3) {
    echo "<ul>";
    foreach($model->visitases as $foreignobj) { 

				printf('<li>%s - %s</li>', 
                                        CHtml::link($foreignobj->fecha, array('visitas/view', 'id' => $foreignobj->id)),
                                        CHtml::link($foreignobj->idClinica0->nombre, array('visitas/view', 'id' => $foreignobj->id))
                                );
                                
				}
    echo "</ul>";
}
?>

